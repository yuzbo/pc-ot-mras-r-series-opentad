# BH-SDC Train Blocker Fix 20260625

Timestamp: 2026-06-25 01:55:40 +08:00

Route label: `DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3`

## Scope

- Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_BHSDC_TrainBlockerFix_Worktree_20260625`
- Branch: `codex/bh-sdc-train-blocker-fix-20260625`
- Base commit: `ae81be06f5228572bd6450619cddb8c8b0fc2fde`
- Remote training, sync, Slurm, and local training were not run in this fix stage.

## Train Blocker

BH-SDC formal training failed before the first loss step during optimizer grouping:

```text
AssertionError: parameters {'token_compressor.refine_scale'} were not separated into either decay/no_decay set!
```

Root cause: `ActionFormer.get_optim_groups` handled biases, Conv/Linear weights, norm weights, relative position encodings, query embeddings, and specific `Scale`/`AffineDropPath` scale parameters, but did not classify BH-SDC bridge scalar learned scales such as `token_compressor.refine_scale`.

## Fix

- `opentad/models/detectors/actionformer.py`: classifies trainable scalar/vector parameters whose local names end with `scale`, `scales`, `gate`, or `gates` into the no-decay optimizer group.
- Conv/Linear weights remain in decay groups.
- Biases, norm weights, relative position encodings, and query/slot embeddings keep their existing no-decay behavior.
- No new non-finite gradient hard gate was added.

## Regression Test

- `tests/test_bh_sdc_actionformer_integration.py`: added `test_bh_sdc_refine_scale_is_in_optimizer_no_decay_group_once`.
- The test opens the BH-SDC sparse-to-dense bridge refine path, calls `get_optim_groups`, and verifies `token_compressor.refine_scale` appears exactly once in the no-decay group.
- Before the fix, this test failed with the same assertion as the formal-training blocker.

## Verification Passed

All verification below was run from the owned worktree and passed:

```powershell
C:\Users\skywalker\.conda\envs\torch_1\python.exe -m py_compile opentad/models/detectors/actionformer.py opentad/models/selectors/bh_sdc_frame_selector.py
```

Result: passed.

```powershell
C:\Users\skywalker\.conda\envs\torch_1\python.exe -m pytest tests/test_bh_sdc_actionformer_integration.py::test_bh_sdc_refine_scale_is_in_optimizer_no_decay_group_once -q -rs
```

Result: `1 passed in 4.24s`.

```powershell
C:\Users\skywalker\.conda\envs\torch_1\python.exe -m pytest tests/test_bh_sdc_actionformer_integration.py tests/test_bh_sdc_core.py tests/test_bh_sdc_config_gate.py tests/test_bh_sdc_launcher_gate.py -q -rs
```

Result: `29 passed in 50.50s`.
