# DIVERGENT BVR-TWB Local Implementation 2026-06-29

Route label: `DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3`

Stage: `LOCAL_IMPLEMENTATION`

This implementation is local-only. It contains CPU validators, synthetic ledger smoke, VOI-BBC selector core, witness packet generation, a trainable value surface with heuristic fallback, dynamic budget control, sparse gather validation, OpenTAD LoadFrames handoff precheck code, and matched controls.

Locked items remain locked: remote sync, Slurm, training, validation/test evaluation, mAP, runtime/FLOPs, deploy claims, paper claims, C3 merge, T2 as a first-version dependency, and dense raw handoff sparse claims.

Implemented components:

- `StateScout`: deterministic deploy-visible curves from actionness and optional motion-like preview signal.
- `Scaffold`: low-cost safety candidates and gap-guard candidates, not a learned novelty claim.
- `BoundaryBelief` and `WitnessPackets`: scout prior brackets, role-labeled witness packets, and selected-witness posterior belief traces.
- `PacketValuePredictor`: heuristic fallback and mock constant ablation with VOI component diagnostics and actionness-component ratio checks.
- `DynamicBudgetController`: marginal VOI/value frontier, posterior belief-risk updates, risk-constrained scaffold/gap handling, active bracket witness coverage, deterministic tie-breaks, and explicit stop reasons.
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
- Value scoring exposes component diagnostics: `belief_width_gain`, `role_gain`, `gap_gain`, `short_action_gain`, `redundancy_repulsion_penalty`, `actionness_component`, and non-action contribution fractions.
- Witness packet gap features receive scaffold/selected positions, and residual `gap_bridge` candidate generation uses the explicit BVR `max_gap` contract.
- StateScout deploy metadata rejection is recursive over nested dict/list structures.
- Deploy ledgers now carry explicit `claim_mode`. The local mode remains `local_gather_smoke`; sparse compute remains locked unless a future `sparse_forward_audit` proves `detector_forward_temporal_len == valid_k < dense_T` and no dense raw handoff.
- Matched controls inherit BVR max-gap by default and record random seed, jitter, per-case uniform overlap, scaffold-only policy, and `twb_no_regret` uniform fallback count/ratio.
- Synthetic summaries add posterior belief width and fallback diagnostics while keeping `claim_status` exactly `local_gather_smoke_only_no_sparse_compute_or_metric_claim`.
- Final-review blocker fix: `belief_width_safe` now requires every active bracket in `active_belief_update_trace` to be updated by selected witness packets and individually `belief_width_safe=True`. Uncovered active brackets are not excluded from the stop decision. `_infer_stop_reason` uses the same all-active-safe semantics.
- Local hardening: deploy ledger validation now rejects forged `belief_width_safe` stops, nested leakage aliases such as `ground_truth`, and synthetic builder output is schema-validated before JSONL write.

Still locked: remote sync, Slurm, training, validation/test evaluation, mAP, runtime/FLOPs, deploy claims, paper claims, C3/combo merge, T2 first-version dependency, and dense raw handoff sparse claims.

## Sparse Forward Handoff Audit Local Update

Implemented the local true-sparse handoff audit gate needed before a later `PRECHECK_ONLY` stage:

- `SparseRawHandoffLedger` records selector, raw decode, backbone pre/post, detector prepad/pad, mask true count, original-time decode, route label, and claim status.
- `SparseForwardAuditContext` provides an observability-only collection surface for fake/module-forward-style evidence. It does not modify tensors or unlock claims.
- `validate_sparse_forward_ledger` fails closed on dense raw handoff, dense backbone chunk count, detector padding counted as valid, missing original-time decode, leakage fields, route mixing, and local sparse-compute claim escalation.
- `tools/bvr_twb/audit_sparse_forward_precheck.py` supports `--mode fake_raw` and `--mode module_fake_forward`, writes JSONL ledgers plus `summary.json`, and keeps `sparse_compute_claim=false`.
- `raw_handoff.py` records sparse raw indices and padded duplicate counts so duplicate edge padding cannot be counted as fresh valid observations.

Local command:

```powershell
python tools/bvr_twb/audit_sparse_forward_precheck.py --mode fake_raw --out-dir .tmp_bvr_twb_sparse_forward_audit --overwrite
```

Expected status after this update:

- Claim status: `sparse_forward_precheck_shape_only_no_metric_claim`.
- Sparse compute claim: `false`.
- No mAP, runtime/FLOPs, deployment readiness, paper claim, remote sync, Slurm, training, validation/test evaluation, or `tools/test.py` is unlocked.
- A later real-data or real-module `PRECHECK_ONLY` still needs review/launch authorization and must prove sparse raw observations through decode, backbone, detector valid mask, and original-time temporal decode.

## First-Trainable BVR-TWB Route Update

The route now has a first-trainable OpenTAD integration surface rather than only fake/local gather evidence:

- `regret_labels.py` implements train-only GT-geometry proxy labels for packet omission regret. The target combines boundary coverage loss, short-action miss risk, bracket width/uncertainty, and gap risk. It explicitly marks labels as `training_only`, `uses_gt=true`, and forbids `val/test/deploy`.
- `trainable_value.py` implements packet feature vectorization, a lightweight MLP value model factory, SmoothL1 value loss, and a learned-value adapter that falls back to the existing heuristic when no trained model is supplied.
- `open_tad_bridge.py` builds deploy-visible scout curves, brackets, witness packets, value scores, dynamic min/max-K controller outputs, selected raw-frame positions, original-time metadata, and optional train-only value labels.
- The fallback `PacketValuePredictor` now records an explicit `actionness_component` and computes actionness/non-action fractions from absolute value-component contributions, closing the prior weak anti-actionness-only diagnostic.
- `LoadFrames` now supports `method="bvr_twb_dynamic_subsample"`. It constructs the dense window through `random_trunc` or `sliding_window`, runs BVR-TWB selection, sets `frame_inds = dense_window[keep_positions]`, writes irregular original-time metadata, keeps selector provenance free of GT/teacher/cache, and records a BVR ledger. It does not count padding duplicates as valid observations.
- `Collect` passes BVR ledger, selected positions, dense valid length, and train-only value labels through `metas`.
- `input_bvr_twb_dynamic_adapter_irregular_headv3.py` is the full-train candidate config. It inherits the irregular Adapter/ActionFormer family, switches train/val/test pipelines to BVR-TWB dynamic subsampling, uses train-only value labels only in train, sets batch size 1 for variable-length raw inputs, and removes dense post-backbone interpolation from the candidate path.
- `audit_opentad_bvr_twb_pipeline.py` is the local/remote PRECHECK_ONLY pipeline audit entrypoint. It runs real `LoadFrames` when torch/OpenTAD is importable and otherwise writes a fail-closed blocked summary without metric/deploy/sparse-compute claims.
- `validate_bvr_twb_launch_gate.py` is a fail-closed local gate. It requires a clean BVR config and an unblocked, fully validated OpenTAD pipeline precheck summary before even requesting remote `PRECHECK_ONLY` review. It never unlocks full train by itself.

Current local evidence:

```powershell
python -m pytest tests/test_bvr_twb_validators.py tests/test_bvr_twb_synthetic_smoke.py tests/test_bvr_twb_matched_controls.py tests/test_bvr_twb_sparse_forward_audit.py tests/test_bvr_twb_opentad_pipeline.py -q
```

Result in the current Windows environment after Round2 hardening: `38 passed, 5 skipped`. The skipped tests are the true OpenTAD/torch pipeline tests because local torch import fails with a Windows DLL initialization error from `torch/lib/c10.dll`.

```powershell
python tools/bvr_twb/audit_opentad_bvr_twb_pipeline.py --out-dir .tmp_bvr_twb_opentad_pipeline_audit --overwrite
```

Current local result: fail-closed blocked summary with `blocked=true`, `all_validated=false`, `sparse_compute_claim=false`, caused by local torch DLL import failure. This is an environment blocker for local real `LoadFrames` execution, not a selector/regret/ledger pass.

```powershell
python tools/bvr_twb/validate_bvr_twb_launch_gate.py --config configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3.py --precheck-summary .tmp_bvr_twb_opentad_pipeline_audit/summary.json
```

Current local result: exit code `1` with `{"gate_pass": false, ...}` because the OpenTAD pipeline audit summary is blocked at `import_torch_or_loadframes`. This is the intended fail-closed state and does not unlock remote `PRECHECK_ONLY`.

Still locked after this update: remote sync, Slurm, training, validation/test evaluation, mAP, runtime/FLOPs, deploy claim, paper claim, C3/combo merge, and true sparse-backbone compute claim. The next required non-local evidence is a torch-valid real one-batch PRECHECK_ONLY run that proves the BVR selected raw frames flow through decode/backbone/detector masks with original-time temporal grids. Adapter dynamic-temporal compatibility remains a specific item for that one-batch gate because the Adapter family has fixed temporal-size assumptions.

Review-gate note: final read-only subagent review was attempted through the available `claude_review` tool twice on 2026-06-29, but both calls failed at the tool/CLI layer with `Claude CLI did not return JSON output`. No subagent review verdict is accepted from those attempts, so deployment/sync/full-train readiness remains locked by review gate as well as by the local torch/OpenTAD precheck blocker.

## VOI-BBC Route Deepening Update

`VOI-BBC` means Value-of-Information Boundary Belief Controller. It is a deeper specification of the same BVR-TWB route, not a C3/combo branch. The selector objective is now explicitly framed as expected localization-risk reduction, posterior entropy/credible-width reduction, and train-only predicted detector regret, rather than actionness top-k, uniform quota, or a plain max-gap engineering rule.

Implemented local mechanisms:

- Boundary belief is an acquisition posterior state. Each bracket trace records before/center/after/short-guard witness evidence, posterior entropy, posterior credible width, risk mass, two-sided witness coverage, and `updated_by_selected_witness`. A `belief_width_safe` stop is rejected unless active brackets were updated by selected witnesses and are individually safe.
- Scaffold/gap anchors are now risk-constrained low-cost candidates and fail-closed safety constraints. The controller ledger records `mandatory_scaffold_first=false`; endpoint safety-floor selections, when used, are explicitly marked as `safety_floor` rather than learned novelty.
- `DynamicBudgetController` uses a marginal VOI frontier. Each row records before/after belief state, belief risk, marginal VOI score, component diagnostics, constraint state, and stop reason. Stop reasons now distinguish `belief_width_safe`, `risk_constraints_satisfied`, `regret_saturation`, `budget_cap`, `candidate_exhausted`, and `gap_guard`.
- Value diagnostics now use VOI-BBC component names: `expected_entropy_reduction`, `expected_width_reduction`, `expected_gap_risk_reduction`, `short_action_value`, `two_sided_witness_value`, `predicted_regret`, `value_per_cost`, and `actionness_component`. The anti-actionness gate is based on absolute component contribution ratio, not only selected-position overlap.
- Train-only regret labels now include VOI targets: entropy reduction, width reduction, boundary-risk reduction, and omission regret. These labels remain GT geometry proxies for `train` only; `val/test/deploy` GT/teacher/cache use remains fail-closed.
- The OpenTAD bridge still performs true raw-frame handoff at `LoadFrames` by setting `frame_inds = dense_window[keep_positions]`, preserving original-time metadata and keeping padding duplicates out of valid observations. The bridge ledger now carries VOI-BBC bracket/controller summaries.
- Validators now fail closed on forged posterior-safe traces, actionness-dominated value components, stop reasons inconsistent with posterior risk/gap state, val/test VOI labels, route mixing, and OpenTAD launch-gate bypass.

Current local VOI-BBC evidence:

```powershell
python -m pytest tests/test_bvr_twb_validators.py tests/test_bvr_twb_synthetic_smoke.py tests/test_bvr_twb_matched_controls.py tests/test_bvr_twb_sparse_forward_audit.py tests/test_bvr_twb_opentad_pipeline.py tests/test_bvr_twb_voi_bbc.py -q
```

Result: `46 passed, 5 skipped`. The skipped tests remain the torch/OpenTAD pipeline tests blocked by local Windows `torch/lib/c10.dll` initialization failure.

```powershell
python tools/bvr_twb/build_synthetic_ledgers.py --out-dir .tmp_bvr_twb_ledgers --overwrite
```

Result after VOI-BBC update: `cases=7`, dynamic `k=[7, 15, 7, 13, 16, 17, 14]`, stops `{'risk_constraints_satisfied': 3, 'belief_width_safe': 4}`, claim status still `local_gather_smoke_only_no_sparse_compute_or_metric_claim`.

Still locked after VOI-BBC update: remote sync, Slurm, training, validation/test evaluation, mAP, runtime/FLOPs, deploy claim, paper claim, C3/combo merge, and true sparse-backbone compute claim. VOI-BBC local tests prove selector/ledger/provenance contracts, not detector mAP or runtime.

## Final Review Blocker Fix: Adapter Bridge And Launch Gate

The final read-only review identified two blockers and both are fixed locally:

- Adapter compatibility: the BVR dynamic valid set can be shorter than the fixed temporal size expected by the ViT Adapter path. `adapter_bridge.py` now provides an explicit `adapter_fixed_length_padded_bridge`. `LoadFrames` feeds fixed-length `frame_inds` to the Adapter by hold-last duplicate padding, while preserving the sparse fresh `valid_k`, `selected_positions`, `selected_frame_inds`, `adapter_valid_raw_mask`, detector mask true count, and original-time metadata. Padding duplicates are recorded as compatibility input only and `adapter_padding_counts_as_valid=false`.
- Claim boundary: this bridge is not true sparse compute evidence. Ledgers carry `adapter_bridge_mode`, `adapter_input_frame_count`, `adapter_padding_duplicate_count`, `detector_mask_len`, and `detector_mask_true_count`. Validators allow padded Adapter input only when these fields prove duplicates are not valid observations and `sparse_compute_claim=false`.
- Launch gate: config text checking now removes only the BVR route label and then rejects `C3`, `C3-Pro`, `C3_PRO`, `GlobalRank`, `COMBO`, and other forbidden route tokens. The gate also requires `adapter_fixed_length_padded_bridge` metadata in both config and successful precheck summary. A local gate pass can only produce `FINAL_READ_ONLY_REVIEW_THEN_LINUX_PRECHECK_ONLY`; full train remains locked.

Focused blocker tests:

```powershell
python -m pytest tests/test_bvr_twb_opentad_pipeline.py -q
```

Result after blocker fixes: `6 passed, 5 skipped`. The skipped cases are torch/OpenTAD import-dependent tests in the current Windows environment; the new pure local tests cover fixed-length Adapter padding, invalid padding masks, and C3/COMBO launch-gate rejection.

Full local BVR test result after blocker fixes:

```powershell
python -m pytest tests/test_bvr_twb_validators.py tests/test_bvr_twb_synthetic_smoke.py tests/test_bvr_twb_matched_controls.py tests/test_bvr_twb_sparse_forward_audit.py tests/test_bvr_twb_opentad_pipeline.py tests/test_bvr_twb_voi_bbc.py -q
```

Result: `49 passed, 5 skipped`. Next allowed stage remains another final read-only review, then a Linux/N16R4 `PRECHECK_ONLY` one-batch audit if review passes. Long training is not unlocked by these local fixes.

## Final Review Blocker Fix Round 2: Raw Positions vs Detector Feature Centers

The second read-only review found that raw selected positions were being reused as detector temporal-grid centers. This is now fixed by separating temporal metadata by layer:

- Raw acquisition metadata is kept in `raw_selected_positions`, `bvr_twb_raw_selected_positions`, and `bvr_twb_raw_selected_valid_len`. These fields describe fresh raw-frame observations for the BVR ledger, Adapter bridge, and BackboneWrapper time embedding.
- Detector/head metadata is kept in `detector_feature_positions`, `bvr_twb_detector_feature_positions`, and `irregular_selected_positions`. These fields are feature/tubelet-level centers and their length equals `detector_feature_valid_k` / detector mask true count, not raw `valid_k`.
- For `feature_stride > 1`, detector centers are computed by grouping real selected raw positions. If the last group is partial, its center is the mean of only real raw positions in that group. Adapter padded duplicate positions are never used as fresh detector centers.
- `BackboneWrapper` now prefers `bvr_twb_raw_selected_positions` for backbone time-embedding metadata, while `IrregularActionFormer` continues to consume `irregular_selected_positions` as detector feature centers. This avoids reusing one key for two temporal granularities.
- `validate_bvr_twb_pipeline_ledger` now requires `raw_selected_positions`, `detector_feature_valid_k`, and `detector_feature_positions`, and recomputes detector centers from raw positions plus `bvr_twb_feature_stride`. It fails closed if detector centers look like raw-position prefixes or include padded duplicates.

Focused round-2 test result:

```powershell
python -m pytest tests/test_bvr_twb_opentad_pipeline.py -q
```

Result: `7 passed, 5 skipped`. The new passing test covers `feature_stride=2`, odd `valid_k`, and verifies the final detector center is a partial-group mean rather than hold-last padding.

Full local BVR result after round-2 blocker fix:

```powershell
python -m pytest tests/test_bvr_twb_validators.py tests/test_bvr_twb_synthetic_smoke.py tests/test_bvr_twb_matched_controls.py tests/test_bvr_twb_sparse_forward_audit.py tests/test_bvr_twb_opentad_pipeline.py tests/test_bvr_twb_voi_bbc.py -q
```

Result: `50 passed, 5 skipped`. Long training remains locked until final read-only review passes and a Linux/N16R4 `PRECHECK_ONLY` one-batch audit validates the real OpenTAD forward path.

## Linux PRECHECK_ONLY Pass Evidence

Remote `PRECHECK_ONLY` now passes for the BVR-TWB / VOI-BBC route on the N16R4 protected hold:

- Route label: `DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3`.
- Local precheck evidence before retry: focused BVR suite reached `51 passed, 5 skipped`, including `tests/test_bvr_twb_opentad_pipeline.py` at `8 passed, 5 skipped`, after fixing the worktree-local output-directory test path while preserving `safe_prepare_output_dir` fail-closed behavior.
- Final read-only review status for this stage: prior read-only review blockers were fixed in the owned worktree; the allowed next action was Linux/N16R4 `PRECHECK_ONLY`, not long training.
- Remote host context: N16R4 protected hold `1118197`, node `g0030`, `CUDA_VISIBLE_DEVICES=1`.
- Remote worktree: `/data/home/sczc063/run/yuzibo/OpenTAD_BVR_TWB_Precheck_20260629_224944`.
- Remote log directory: `/data/home/sczc063/run/yuzibo/OpenTAD_BVR_TWB_Precheck_20260629_224944/logs/bvr_twb_precheck/precheck_gpu1_20260629_231853`.
- Remote tests: `56 passed in 18.34s`.
- OpenTAD pipeline audit: `ledgers=3`, `all_validated=True`, `sparse_compute_claim=False`, `blocked=False`.
- Launch gate summary: `{"adapter_bridge_mode":"adapter_fixed_length_padded_bridge","gate_pass":true,"full_train_unlocked":false,"remote_sync_unlocked_by_local_gate":false,"route_label":"DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3","sparse_compute_claim":false}`.
- PASS marker: `PASS_BVR_TWB_LINUX_PRECHECK_ONLY`.

Claim and launch boundary after this pass:

- Linux `PRECHECK_ONLY` is passed for import, BVR tests, real `LoadFrames` pipeline audit, adapter fixed-length padded bridge metadata, and fail-closed launch-gate behavior.
- The Adapter compatibility bridge remains explicitly `adapter_fixed_length_padded_bridge`; it is not a true sparse-compute claim.
- Full training, Slurm long run, validation/test evaluation, mAP, runtime/FLOPs, deploy readiness, paper claims, and true sparse-backbone/detector sparse-compute claims remain locked.
- Unlocking full train still requires the formal full-train gate, including the required review/Pro/explicit override path for this route stage.

## Train Smoke Blocker Fix: Adapter Chunk Positional Embedding

The first N16R4 train smoke after the optimizer fix reached the first training batch, then failed in `vit_adapter.py` at `x = x + pos_embed` with a token-count mismatch: `9600` actual tokens versus `800` positional-embedding tokens. Root cause: the BVR config had replaced the inherited Adapter custom pipeline with `_delete_=True` and removed the original 16-frame chunking pre-processing. VideoMAE's positional embedding is defined for one 16-frame clip (`8` tubelet tokens times `10x10` spatial tokens = `800`), while the BVR candidate was sending the whole fixed-length Adapter bridge window directly into one VideoMAE forward.

The route fix restores the Adapter-required chunk contract without restoring dense temporal interpolation:

- `input_bvr_twb_dynamic_adapter_irregular_headv3.py` now defines `chunk_num = window_size * scale_factor // 16`.
- The backbone custom config restores `pre_processing_pipeline` with `b n c (t1 t) h w -> (b t1) n c t h w`, so the Adapter sees 16-frame chunks and `pos_embed` stays aligned.
- The `post_processing_pipeline` keeps `Reduce` and adds `(b t1) c t -> b c (t1 t)` to reassemble tubelet features for the detector.
- Dense `Interpolate` remains absent. The detector sees the tubelet feature sequence produced from selected/padded Adapter input, and BVR ledgers still mark `sparse_compute_claim=false`.
- The fix does not change `LoadFrames` selection: BVR still selects sparse raw frames before `DecordDecode`; hold-last padding remains explicit Adapter compatibility input and does not count as fresh valid observations.

Focused local regression now asserts that the BVR config keeps chunking, reassembles chunks, and does not reintroduce dense interpolation. This unlocks another read-only review / remote train-smoke retry, not formal full train, mAP, runtime/FLOPs, deploy, paper, or true sparse-compute claims.

## One-Epoch Diagnostic Smoke Evidence

After the chunk positional-embedding fix, the N16R4 GPU1 diagnostic smoke advanced through one diagnostic epoch:

- Remote worktree: `/data/home/sczc063/run/yuzibo/OpenTAD_BVR_TWB_Precheck_20260629_224944`.
- Remote log directory: `/data/home/sczc063/run/yuzibo/OpenTAD_BVR_TWB_Precheck_20260629_224944/logs/bvr_twb_train_smoke/smoke_gpu1_20260629_235750_chunk_fix`.
- Protected hold: Slurm job `1118197`, node `g0030`, `CUDA_VISIBLE_DEVICES=1`; parent hold was not released.
- Remote tests before smoke: `60 passed in 22.48s`.
- Result: one diagnostic epoch reached `[000][00199/00199]` and `Training Over...`.
- Cleared blocker: `POS_EMBED_COUNT=0`, `RUNTIME_COUNT=0`, and no `9600 vs 800` positional-embedding crash.
- Caveat: `NONFINITE_COUNT=3`; non-finite gradient skip diagnostics occurred in `rpn_head.reg_head.weight`. This did not crash the diagnostic smoke, but it remains a formal-full-train risk to inspect before unlocking long training.

This smoke is execution evidence only. It does not unlock mAP, runtime/FLOPs, deploy readiness, paper claims, or true sparse-compute claims. Full train remains locked until the coordinator / Pro / explicit user gate allows it for this route stage.
