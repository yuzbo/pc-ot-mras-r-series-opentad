# DIVERGENT BVR-TWB Launch Decision 2026-06-30

Route label: `DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3`

Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_DivergentRoutes_ProReview_Worktree_20260630`

Owned branch for this documentation stage: `codex/divergent-routes-pro-review-20260630`

Single-owner statement: for this launch-decision documentation stage, this document is the only writable artifact. No code, config, test, validator, launcher, git index, branch, remote, Slurm, training, evaluation, shared worktree, ABR, MDL, C3, or combo-route file was modified by this decision record.

Decision timestamp: `2026-06-30T05:00:25+08:00` coordinator evidence time, recorded locally on `2026-06-30` Asia/Shanghai.

## Verdict

`NO_GO_FOR_NEW_BVR_LAUNCH_NOW`

The BVR final branch cannot start a new Slurm/full-train launch right now.

This is not a technical rejection of the final BVR gate code. The current no-go is a launch-decision boundary:

- GPU1 is already occupied by an older BVR run from branch/commit `cf662dd`, not the final BVR gate branch/commit.
- The valid GPT-5.5 Pro decision allowed only `PASS_ALLOW_BVR_FULL_TRAIN_GATE_ONLY`, meaning gate-packet/review evidence only.
- The BVR gate/precheck evidence still reports no training, no metric claim, no sparse-compute claim, and no direct Slurm/train/test/evaluation unlock.
- A fresh coordinator/user go decision is still required after resource state is clear and the exact final branch/commit/resource plan is restated.

## Evidence Used

Local files inspected in the owned worktree:

- `research-wiki/experiments/DIVERGENT_BVR_TWB_FULL_TRAIN_GATE_PACKET_20260630.md`
- `logs/divergent_routes_abbr_mdl_bvr_rosetta_pro_followup_20260630.txt`
- `docs/DIVERGENT_BVR_TWB_LOCAL_IMPLEMENTATION_20260629.md`
- `configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3.py`
- `tools/bvr_twb/validate_bvr_twb_launch_gate.py`

Authoritative coordinator-provided remote evidence at `2026-06-30T05:00:25+08:00`:

- Protected parent hold `1118197 pcot_dbg2g` is running on `g0030`; it must not be released, cancelled, modified, replaced, or allowed to self-terminate by this decision.
- `parajobs` showed GPU1 at about `59%` utilization and `2441 MiB` used, while GPU0 was idle at that instant.
- Slurm step `1118197.356` has been running since `2026-06-30T03:27:51`; job name `bash`; TRES step shows `gres/gpu:2`; SrunHost PID `ln01:3183070`.
- The old BVR run log `/data/home/sczc063/run/yuzibo/OpenTAD_BVR_TWB_StabilityFix_20260630_9b15fe3/logs/bvr_twb_formal_cf662dd_20260630_0329_gpu1_relaunch/train.log` shows finite losses through epoch 35 ending `04:59:36`, then epoch 36 started.
- That running evidence belongs to old branch/commit `cf662dd`; it is not final gate branch/evidence `cbef02b` / `5697655`.
- Final BVR precheck summary from `/data/home/sczc063/run/yuzibo/OpenTAD_BVR_TWB_Final_20260630_5697655/logs/precheck_20260630_bvr_5697655/bvr_twb_audit_v3/summary.json` reports:
  - `all_validated=true`
  - `deterministic_preview_fallback_used=false`
  - `preview_sources=[deploy_visible_metadata_actionness]`
  - `scout_sources=[deploy_visible_raw_or_metadata_scout]`
  - `value_modes=[deploy_heuristic_voi]`
  - `no_training=true`
  - `no_metric_claim=true`
  - `sparse_compute_claim=false`

Pro/gate evidence boundary:

- GPT-5.5 Pro follow-up verdict: `PASS_ALLOW_BVR_FULL_TRAIN_GATE_ONLY`.
- Accepted meaning: BVR-TWB / VOI-BBC may enter a formal full-train gate record/review stage.
- Explicit non-meaning: no Slurm, no `tools/train.py`, no `tools/test.py`, no validation/test evaluation, no detector mAP, no runtime/FLOPs, no deploy claim, no paper claim, no C3/combo merge, and no protected-hold change.
- The gate packet records final route evidence branch `codex/divergent-bvr-twb-final-20260630`, final repair/precheck commit `569765513992914fa4a02982505692dd9681c266`, earlier repair commit `393e890fca4d046538d7a204f70e226f875817bf`, and evidence HEAD `cbef02b`.

## Launch Decision

Current decision: `NO_GO_FOR_NEW_BVR_LAUNCH_NOW`.

Reasoning:

1. The only running BVR long evidence at the decision time is old `cf662dd` evidence. It may be useful as historical stability context, but it cannot authorize a new final-branch launch and cannot be reported as final `cbef02b` / `5697655` training evidence.

2. GPU1 is not clear. It is currently tied to protected hold `1118197` and running step `1118197.356`, with the old BVR process active. Starting another GPU1 BVR launch now would create resource conflict and attribution confusion.

3. Pro did not grant direct Slurm or full-train permission. Its accepted decision was gate-only: `PASS_ALLOW_BVR_FULL_TRAIN_GATE_ONLY`.

4. The final precheck summary is favorable but explicitly non-training evidence. It says the final path is validated for precheck, has deploy-visible metadata actionness, deploy-visible raw/metadata scout, `deploy_heuristic_voi`, no diagnostic preview fallback, no training, no metric claim, and no sparse-compute claim.

Therefore, the BVR final branch is technically closer to launch readiness, but the correct launch decision at this moment is no new BVR Slurm/full-train launch.

## Required Next Condition

A later BVR launch can be reconsidered only after all of the following are true and recorded by the coordinator/user as a fresh go decision:

1. Resource state is clear for the intended GPU plan. In particular, GPU1 use must be explicitly authorized after the old `cf662dd` run/step is resolved or a non-conflicting plan is named. Protected parent hold `1118197 pcot_dbg2g` must remain untouched unless the user explicitly authorizes an exact hold action.

2. The launch target is explicitly restated as the final BVR branch/evidence, not the old run:
   - final BVR branch: `codex/divergent-bvr-twb-final-20260630`
   - final precheck clone/commit: `569765513992914fa4a02982505692dd9681c266`
   - evidence HEAD/gate packet lineage: `cbef02b`
   - route label: `DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3`
   - old running branch/commit `cf662dd` must not be reused as final gate evidence.

3. The exact final config remains `configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3.py` with:
   - `method="bvr_twb_dynamic_subsample"`
   - `bvr_twb_scout_source="deploy_visible_raw_or_metadata_scout"`
   - `bvr_twb_require_deploy_visible_scout=True`
   - `bvr_twb_allow_diagnostic_preview_fallback=False`
   - no `diagnostic_deterministic_preview`
   - `bvr_twb_value_mode="deploy_heuristic_voi"`
   - `bvr_twb_train_value_labels=True` only in train
   - `bvr_twb_train_value_labels=False` in val/test
   - `adapter_fixed_length_padded_bridge`
   - `solver.amp=False`
   - `solver.fp16_compress=False`
   - `filter_invalid_regression_samples=False`

4. The exact final precheck summary remains consistent with the authoritative summary:
   - `all_validated=true`
   - `blocked=false`
   - `deterministic_preview_fallback_used=false`
   - `preview_sources` nonempty and limited to formal deploy-visible preview sources
   - `scout_sources` nonempty and deploy-visible
   - `value_modes` exactly `["deploy_heuristic_voi"]` as a set
   - `value_labels_used_at_test=false` or absent/false
   - `no_training=true`
   - `no_metric_claim=true`
   - `sparse_compute_claim=false`
   - adapter padding duplicates do not count as valid observations

5. There is still no GT, teacher, raw prediction, prediction cache, evaluator, post-processing, oracle boundary, oracle residual, validation/test teacher leakage, or hidden dense raw-backbone handoff in selector/provenance/evaluation-time behavior.

6. The next command/resource plan is explicit and bounded before launch: target clone path, exact branch/commit, GPU id, parent hold policy, log directory, expected command, monitoring cadence, and stop/continue rule.

7. A fresh coordinator/user decision says the launch may proceed. The existing Pro verdict and precheck do not by themselves authorize the launch.

## Still Locked

The following remain locked by this decision:

- Any write outside `research-wiki/experiments/DIVERGENT_BVR_TWB_LAUNCH_DECISION_20260630.md`.
- Any code/config/test/validator/launcher modification.
- Any ABR, MDL, C3, C3-Pro, combo-route, or shared/main worktree modification.
- `git add`, commit, push, branch switch, branch creation, or index manipulation.
- SSH, remote sync, Slurm, `srun`, `sbatch`, training, evaluation, `tools/train.py`, and `tools/test.py`.
- Protected hold `1118197 pcot_dbg2g` release, cancellation, replacement, modification, or self-termination policy change.
- New BVR GPU launch before the fresh resource-clear and coordinator/user go decision.
- Detector mAP, high-IoU, runtime/FLOPs, sparse-compute, deployment, or paper claims.
- Treating old `cf662dd` run progress as final `cbef02b` / `5697655` full-train evidence.
- Claiming raw-RGB scout overhead, detector accuracy, deployment readiness, or final training stability has been validated.

## Risks

- Resource/attribution risk: GPU1 is occupied by an old `cf662dd` BVR run. Launching a new final-branch BVR job now would mix old-run and final-gate evidence.
- Decision-scope risk: Pro approved gate-only evidence, not direct Slurm/full-train execution.
- Evidence-boundary risk: the final summary is a precheck summary with `no_training=true` and `no_metric_claim=true`; it cannot support mAP, runtime, FLOPs, deploy, paper, or sparse-compute claims.
- Scout-cost risk: deploy-visible metadata or low-resolution raw-RGB scout is protocol-valid, but overhead and detector impact are not yet quantified.
- Stability risk: old `cf662dd` finite-loss evidence through epoch 35 is encouraging but not final-branch proof. Any final-branch run still needs its own logs and stop/continue evidence.
- Protocol risk: any reintroduction of diagnostic deterministic preview, learned value without explicit loaded model, train-only labels outside train, hidden dense handoff, C3/combo tokens, or padding counted as valid observations is a hard no-go.
