# C3 PQR RankCal V1 GPT-5.5 Pro Launch-Decision Gate Prompt - 2026-06-30

请用中文做只读 launch-decision / implementation-review。不要建议启动训练；本轮只判断下一步是否允许 2-iter runtime smoke、8-epoch short diagnostic，以及 formal/full train 是否仍锁定。

## GitHub Evidence

- Repository: https://github.com/yuzbo/pc-ot-mras-r-series-opentad
- Branch: https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/c3-pqr-rankcal-v1-20260629
- Commit: https://github.com/yuzbo/pc-ot-mras-r-series-opentad/commit/dca62cc24c0bef50e53135709820b6a66afe1422
- Commit message: `C3_PQR_RankCalV1 PRECHECK_ONLY implementation evidence`

Please inspect the GitHub branch/commit directly if available. If you cannot inspect GitHub, state that explicitly in `Context verdict` and treat the decision as context-incomplete.

## Route And Objective

- Route label: `C3_MAINLINE_OPTIMIZATION`
- Route family: `C3_ORIGINAL_OPTIMIZATION_ROUTE`
- Variant: `C3_PQR_RankCalV1_MaxIoU`
- Original route purpose: improve the original C3 / Adapter + ActionFormer line through sparse-head proposal ranking calibration. PQR is a detector-head quality/ranking calibration route, not a CADF selector/input-sampling route.
- Current stage: `PRECHECK_ONLY_LIMITED` passed remotely, but direct short diagnostic and formal training remain locked pending this GPT-5.5 Pro gate.

## PVR-QC Background

Prior Adapter quality/reranking work (`PVR-QC` / quality-rescore family) used the existing `AnchorFreeHead` quality-head path to calibrate proposal scores. That family was useful as a ranking-calibration mechanism, but this PQR branch is intentionally narrowed:

- It does not introduce or modify CADF selector logic.
- It does not claim dynamic acquisition or input-side gains.
- It reuses the existing quality-head mechanism in `AnchorFreeHead` with a `max_iou` target and low score-fusion alpha.
- It is intended to test whether proposal-quality calibration can improve sparse Adapter + ActionFormer ranking without changing post-processing, evaluator behavior, or input selection.

## Changed Surface

This commit adds only:

- `configs/adatad/thumos/c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_precheck.py`
- `configs/adatad/thumos/c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_shortdiag.py`
- `configs/adatad/thumos/c3_indirect_original_adatad_32px_a_exact_uniform_backend_control_pqr_rankcal_v1_shortdiag.py`
- `tools/validate_c3_pqr_rankcal_v1_config.py`
- `tests/test_c3_pqr_rankcal_v1_config.py`
- `tests/test_c3_pqr_rankcal_v1_quality_head.py`
- `research-wiki/experiments/C3_PQR_RANKCAL_V1_IMPLEMENTATION_20260629.md`

No model implementation file is changed in this commit. The configs reuse the existing `AnchorFreeHead` quality-head support already present in the base snapshot. The PQR changed surface is detector-head ranking calibration via `model.rpn_head.quality_head_cfg`, plus validator/tests/documentation.

Important config intent:

- Main PQR configs inherit from `input_random_fixed_50pct_adapter.py`.
- The exact-uniform backend control is a stride-2 uniform Adapter backend control, not a selector quota experiment.
- Quality config: `target_mode="max_iou"`, `loss_weight=0.03`, `score_alpha=0.10`, neutral init `weight_init=0.0`, `bias_init=4.59511985013459`.
- `pqr_rankcal_v1.remote_launch_locked=True` remains set in configs.
- `diagnostic_only=True`, `claim_map_improvement=False`, `official_map_claim=False`.

## Explicit Boundaries

Do not merge this with, attribute this to, or evaluate it as:

- CADF selector
- CADF input sampler
- BH-SDC
- any `DIVERGENT_INNOVATION_*`
- dynamic-budget acquisition
- physical-time post-processing
- evaluator or NMS modification
- teacher/cache/raw-prediction shortcut

Future PQR results must not be attributed to a CADF selector. PQR is sparse head/ranking calibration only.

## PRECHECK R2 Evidence

Remote PRECHECK_ONLY evidence from R2:

- Remote log: `/data/home/sczc063/run/yuzibo/OpenTAD_C3PQRRankCal_Precheck_20260630/logs/c3_pqr_rankcal_v1_precheck_r2_20260630_005059_CST.log`
- `py_compile`: PASS
- Three validators: all printed `PASS_C3_PQR_RANKCAL_V1_CONFIG`
- Focused pytest: `16 passed in 15.44s`
- Current gate conclusion: `PASS_PRECHECK_ONLY_LIMITED`
- Current lock: `NO_GO_DIRECT_SHORTDIAG_OR_FORMAL_TRAIN`; next allowed step before any runtime experiment is this Pro gate.

Known implementation note: local Windows torch-backed tests were skipped due to `c10.dll` import failure, but the same focused pytest suite passed remotely on Linux after the test fixture fix.

## Review Questions

Please answer in Chinese with these exact sections:

1. `Context verdict`
   - Did you inspect the GitHub branch/commit? Which files/materials were visible?
   - Is the context sufficient for a launch-decision gate?
2. `Inspected materials`
   - List the code/config/test/doc files inspected.
3. `Verdict`
   - Use one of: `PASS_ALLOW_2ITER_SMOKE_ONLY`, `PASS_ALLOW_8EPOCH_SHORTDIAG`, `FIX_BEFORE_RUNTIME`, `CONTEXT_INSUFFICIENT_GITHUB_NOT_INSPECTED`, `REJECT_ROUTE_DRIFT_OR_LEAKAGE`.
4. `Blocking findings`
   - Focus on leakage, route drift, hidden model/config mismatch, unconsumed config keys, postprocess/evaluator shortcut, NaN/ranking-collapse risk, and launch-gate ambiguity.
5. `Non-blocking findings`
   - Include improvements that should not block the next allowed runtime tier.
6. `Required fixes`
   - State exact files/areas and whether fixes are required before 2-iter smoke, before 8-epoch short diagnostic, or before formal/full training.
7. `2-iter runtime smoke allowed?`
   - Answer yes/no and required command scope if yes. It must remain diagnostic only, no metric claim.
8. `8-epoch short diagnostic allowed?`
   - Answer yes/no and conditions. It must remain diagnostic only unless you explicitly allow otherwise.
9. `Formal/full train remains locked?`
   - Answer yes/no. Default should remain locked unless you have strong code-grounded reason to unlock.
10. `Hidden NaN/ranking/postprocess risks`
   - Specifically assess whether `max_iou` quality targets, low alpha fusion, quality loss normalization, neutral init, `max_seg_num=2000`, and unchanged post-processing could produce hidden NaNs, ranking collapse, score-scale mismatch, or invalid attribution.

Please be strict: transport/context failure is not a technical rejection, but it is not a valid approval. Do not approve formal/full training unless the evidence is enough for that tier. Do not attribute any possible gain to CADF or input selection.
