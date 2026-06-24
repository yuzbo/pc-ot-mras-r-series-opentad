# BH-SDC Original Route Purpose And Drift Check Packet

Date: 2026-06-24

This document is part of the BH-SDC implementation review payload. Future Pro-model review should inspect the GitHub branch/commit directly instead of a manual zip package, and must evaluate both code correctness and route fidelity.

Route label: `DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3`.

Do not merge this route with the existing C3/C3-Pro family unless a future explicit `COMBO_ROUTE_APPROVED` gate exists. The C3/C3-Pro family is labeled `C3_ORIGINAL_OPTIMIZATION_ROUTE` and covers fixed-budget pre-backbone reader/selector optimization, CNN/Motion/Hybrid C3 readers, interval/rank/global-ST/physical-grid C3-Pro candidates, and C3 post-train diagnostics. Pro must reject the review as route-incomplete if the implementation is described as a natural continuation of C3-Pro, interval packet, dynamic-budget guard, or physical-grid ActionFormer rather than as this divergent BH-SDC route.

## Review Delivery Rule

- Preferred review payload: GitHub repository URL, branch name, commit SHA, and exact file list.
- Zip or manual code package is allowed only as a documented fallback when GitHub access is unavailable.
- Claude and Gemini review routes are cancelled for this repository unless the user explicitly asks for a one-off advisory check.
- A Pro review is incomplete if it only reviews code mechanics and does not answer whether the implementation still matches the original route purpose.

## Original Pro Route Source

The original Pro divergence discussion selected a route named:

**BH-SDC: Boundary-Hazard Sparse-to-Dense Acquisition for Temporal Action Detection.**

The route was chosen as a larger departure from fixed 384/768 reader tuning. Its goal is not another uniform-like or slot-query selector. It reframes the task as deployable dynamic temporal acquisition:

- decide how many raw frames/snippets/tokens each video or window should process;
- acquire more evidence near action boundaries, ambiguous motion, and difficult regions;
- acquire fewer inputs for easy, redundant, or background-dominated spans;
- preserve real temporal geometry for high-IoU localization;
- reconstruct or condition dense detector inputs from irregular sparse observations without using test-time ground truth, teacher caches, or raw detector predictions.

## Original Mechanism

The intended BH-SDC mechanism has four parts:

1. **Temporal scout:** a lightweight temporal module estimates actionness, start hazard, end hazard, difficulty, uncertainty, and redundancy from pre-backbone frame features.
2. **Dynamic budget controller:** each sample receives a budget between a configured minimum and maximum, with a target budget used as the neutral operating point.
3. **Boundary-hazard acquisition policy:** selected indices should combine boundary-hazard peaks, actionness/difficulty/uncertainty evidence, and coverage constraints so long gaps do not destroy detector context.
4. **Sparse-to-dense bridge:** the detector still receives a fixed dense temporal lattice for this first implementation, but it is reconstructed from sparse observations and conditioned by observed masks, dense positions, sparse positions, and confidence.

This implementation is a full candidate model for the route, but the launch gate remains locked until Pro verifies alignment and safety after commit-based review.

## Protocol Boundaries

- No validation/test ground truth is allowed in selection, budgeting, reconstruction, post-processing, or runtime metadata.
- No validation/test teacher scores, raw detector predictions, cached oracle decisions, or hidden result shortcuts are allowed.
- Train-time supervision diagnostics may be added later only if the split boundary and deploy-time inputs are explicitly documented.
- Fixed dense output length is a first controlled implementation target, not the final research goal.

## Current Implementation Entry Points

- `opentad/models/selectors/bh_sdc_frame_selector.py`
- `opentad/models/selectors/__init__.py`
- `configs/adatad/thumos/bh_sdc_boundary_hazard_sparse_dense_local_precheck.py`
- `configs/adatad/thumos/bh_sdc_boundary_hazard_sparse_dense_full_train_candidate_n16r4.py`
- `tools/bata/validate_bh_sdc_full_train_gate.py`
- `scripts/run_bh_sdc_full_train_n16r4.sbatch`
- `tests/test_bh_sdc_core.py`
- `tests/test_bh_sdc_config_gate.py`

## Accepted Pro R1 Findings And Fixes

The first Pro review returned `FIX_BEFORE_SUBAGENT_REVIEW`. The accepted blockers were:

1. The sbatch script could be unlocked by environment variables. It now refuses long training unconditionally in this candidate state.
2. The validator used static text checks. It now resolves the config with `mmengine.config.Config.fromfile`.
3. The neutral budget mapping did not prove that `target_budget` was actually used. The controller test now asserts neutral scores map to `target_budget`.
4. Leakage protection was mostly declarative. Selector and bridge now reject forbidden runtime metadata keys in test/validation mode.
5. Tests did not cover real registry build paths or observed-frame preservation. The test suite now covers config buildability and exact preservation at observed dense positions.

## Local Verification Evidence

The following checks passed before preparing the GitHub review handoff:

```powershell
conda run -n torch_1 python -m pytest -q tests\test_bh_sdc_core.py tests\test_bh_sdc_config_gate.py
```

Result: `8 passed`.

```powershell
conda run -n torch_1 python -m pytest -q tests\test_pc_ot_mras_prebackbone_contracts.py::test_actionformer_calls_frame_selector_before_backbone_projection_reader_and_head
```

Result: `1 passed`.

```powershell
python tools\bata\validate_bh_sdc_full_train_gate.py configs\adatad\thumos\bh_sdc_boundary_hazard_sparse_dense_full_train_candidate_n16r4.py --json
```

Result: PASS with locked launch state:

- `allow_long_training`: `false`
- `launch_gate_passed`: `false`
- `dense_window_size`: `768`
- `min_budget`: `256`
- `target_budget`: `384`
- `max_budget`: `448`

```powershell
python -m py_compile opentad\models\selectors\bh_sdc_frame_selector.py tools\bata\validate_bh_sdc_full_train_gate.py tests\test_bh_sdc_core.py tests\test_bh_sdc_config_gate.py
git diff --check
```

Result: passed; `git diff --check` only reported expected line-ending warnings.

## Required Pro Drift-Check Questions

The next Pro review must answer these route-level questions in addition to line-by-line code review:

1. Does the current implementation still match the original BH-SDC purpose, or has it drifted back into a fixed-budget reader/selector variant?
2. Does every entry point consistently carry the route label `DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3` and avoid presenting BH-SDC as a C3/C3-Pro continuation?
3. Does the temporal scout actually provide boundary-hazard and difficulty signals that can support dynamic acquisition, or is it only a cosmetic scoring head?
4. Does the budget controller implement a meaningful dynamic policy with a neutral target budget, or does it collapse into a fixed 384-frame method?
5. Does the sparse-to-dense bridge preserve enough irregular temporal geometry for high-IoU localization, or does dense interpolation erase the intended benefit?
6. Are the leakage gates sufficient for a deployable no-GT-at-test protocol?
7. Is the locked full-train gate appropriate until route alignment is verified?
8. What minimum changes are required before allowing N16R4 long training?

Expected Pro answer schema:

- `Context verdict`
- `Inspected GitHub branch/commit and files`
- `Original-route alignment verdict`
- `Implementation-drift findings`
- `Blocking findings`
- `Non-blocking findings`
- `Required fixes or next experiments`
- `Accepted launch/sync/Slurm decision`
