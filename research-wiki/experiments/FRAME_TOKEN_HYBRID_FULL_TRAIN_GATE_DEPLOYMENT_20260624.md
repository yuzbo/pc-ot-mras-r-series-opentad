# Frame/Token Hybrid Full-Train Gate Deployment

Route label:
`DIVERGENT_INNOVATION_FRAME_TOKEN_HYBRID_DO_NOT_MERGE_WITH_C3`

This report is scoped only to the divergent Frame/Token Hybrid route. It is not
C3, BH-SDC, Event, Boundary, or a combo route.

## Local Full-Train Gate Implementation

Timestamp: 2026-06-24T21:05:00+08:00

Owner scope:

- Owned local worktree:
  `E:/DeskTop/TAD/temrefuse-tad/OpenTAD_FrameToken_PrecheckDeploy_Worktree_20260624`
- Branch: `codex/frame-token-precheck-deploy-20260624`
- Base before this gate implementation:
  `7dcb6aec5fccab0b022df7d018c70853b450a7cf`
- Shared repo was not written.
- Pro/Claude/Gemini review or transport was not used.

Changed surface:

- `tools/bata/validate_frame_token_hybrid_gate.py`
  - Added explicit gate actions: `precheck` and `full-train`.
  - `precheck` remains bound to `ALLOW_FRAME_TOKEN_HYBRID_PRECHECK_ONLY`.
  - `full-train` is bound to `ALLOW_FRAME_TOKEN_HYBRID_FULL_TRAIN`.
  - Full-train gate requires exact active manifest SHA256, resolved config
    SHA256, RUN_TAG, route label, user override statement, coordinator override
    statement, and false GT/teacher/oracle/raw-cache/metric/paper flags.
- `scripts/run_frame_token_hybrid_acquisition_full_train_n16r4.sbatch`
  - New fail-closed N16R4 full-train launcher.
  - Defaults to `PRECHECK_ONLY=1` and exits before training.
  - With `PRECHECK_ONLY=0`, requires
    `ALLOW_FRAME_TOKEN_HYBRID_FULL_TRAIN=1`,
    `FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_JSON`, and
    `FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_SHA256` before running
    `tools/train.py`.
- `scripts/run_frame_token_hybrid_acquisition_precheck_n16r4.sbatch`
  - Updated default route-owned repo path to
    `OpenTAD_FrameToken_PrecheckDeploy_20260624_ecb7c27`.
- `tests/test_frame_token_hybrid_config_gate.py`
  - Added missing full-train gate JSON fail-closed coverage.
  - Added valid full-train gate JSON authorization coverage.
  - Added wrong RUN_TAG and missing override rejection coverage.
  - Added full-train launcher default-lock and gate-binding coverage.
- `docs/en/frame_token_hybrid_acquisition_route_review_context_20260624.md`
  - Updated the route context from precheck-only to default-locked plus
    external full-train gate authorization semantics.

Protocol and leakage status:

- Config default remains fail-closed:
  `ALLOW_FRAME_TOKEN_HYBRID_PRECHECK_ONLY`, `allow_full_train=False`,
  `allow_tools_train=False`, `allow_slurm=False`, `allow_gpu=False`.
- Full train is still locked by default.
- Full train can open only via the external full-train gate JSON plus matching
  SHA256 inputs and `ALLOW_FRAME_TOKEN_HYBRID_FULL_TRAIN=1`.
- No GT/test teacher/oracle/raw-prediction-cache shortcut is allowed by the
  gate.
- Metric and paper claims remain locked.

Local verification:

```text
python tools/bata/validate_frame_token_hybrid_gate.py configs/adatad/thumos/frame_token_hybrid_acquisition_full_train_candidate_n16r4.py --json
PASS: emitted fail-closed precheck-only config JSON.

python -m py_compile tools/bata/validate_frame_token_hybrid_gate.py configs/adatad/thumos/frame_token_hybrid_acquisition_local_precheck.py configs/adatad/thumos/frame_token_hybrid_acquisition_full_train_candidate_n16r4.py tests/test_frame_token_hybrid_config_gate.py opentad/models/selectors/frame_token_hybrid_acquisition_route.py
PASS.

python -m pytest tests/test_frame_token_hybrid_acquisition_route.py tests/test_frame_token_hybrid_metadata_contract.py tests/test_frame_token_hybrid_actionformer_integration.py tests/test_frame_token_hybrid_config_gate.py -q
PASS: 13 passed, 15 skipped in 24.32s.

bash -n scripts/run_frame_token_hybrid_acquisition_precheck_n16r4.sbatch scripts/run_frame_token_hybrid_acquisition_full_train_n16r4.sbatch
PASS.
```

Launch decision after local gate:

- Local gate status: `PASS`.
- Allowed next action: deploy route-owned repo update, generate a fixed RUN_TAG
  precheck manifest and matching full-train gate JSON on N16R4, validate both
  gate actions, and submit one Slurm full-train candidate if validation passes.
