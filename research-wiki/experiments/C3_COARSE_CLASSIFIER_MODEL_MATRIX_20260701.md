# C3 Coarse Classifier Model Matrix 20260701

## Purpose

This is a C3 mainline optimization route, not a divergent innovation route.
The goal is to broaden the pre-backbone coarse action/background classifier
search beyond MobileNet and small TCN variants.

The classifier is diagnostic-only at this stage:

- no detector training;
- no detector evaluation;
- no official mAP claim;
- no test-time GT, teacher, or raw prediction cache;
- no BH-SDC or `DIVERGENT_INNOVATION_*` mixing.

Each candidate should be judged by:

1. frame/snippet action-background quality: AP, AUC, best F1, balanced accuracy;
2. indirect frame-selection quality from `p_action`: `|delta p_action|`,
   uncertainty, transition score, boundary support, action coverage, and
   max-gap;
3. oracle-style Chinese sample visualizations showing `p_action`,
   transition score, GT action intervals, GT boundaries, and selected points.

## First-Wave Download Matrix

The first wave contains models that the current N16R4 OpenTAD environment can
construct directly through installed libraries: `timm`, `torchvision`, and
`pytorchvideo`, plus one manageable Hugging Face VideoMAE-small teacher
snapshot.

| Family | Candidate ids | Purpose |
| --- | --- | --- |
| image backbone + temporal head | `timm_mobilenetv3_large_100_tsm_tcn`, `timm_tf_efficientnetv2_b0_tcn`, `timm_convnext_tiny_tcn`, `timm_resnet18_tcn`, `timm_vit_tiny_patch16_224_temporal` | Test whether stronger per-frame visual recognition plus a temporal head fixes weak coarse `p_action`. |
| native video classifier | `torchvision_r3d_18`, `torchvision_r2plus1d_18`, `torchvision_mc3_18`, `torchvision_s3d`, `torchvision_mvit_v2_s`, `torchvision_swin3d_t` | Test true short-clip video recognition and spatiotemporal context. |
| efficient video classifier | `pytorchvideo_x3d_xs`, `pytorchvideo_x3d_s` | Test deployable low-cost video models. |
| classic/heavier video classifier | `pytorchvideo_c2d_r50`, `pytorchvideo_i3d_r50`, `pytorchvideo_slowfast_r50` | Separate spatial-only, 3D, and SlowFast motion-sensitive behavior. |
| video transformer teacher | `hf_videomae_small_kinetics` | Manageable pretrained video Transformer teacher/upper-bound candidate. |

## Second-Wave Optional Matrix

The second wave is intentionally not downloaded by default because it is heavier
or needs additional adapter work:

- `hf_videomae_base_kinetics`;
- `torchvision_swin3d_s`.

These are teacher/upper-bound candidates, not immediate deployable selectors.

## Implemented Artifacts

- Matrix/download script:
  `tools/bata/c3_coarse_classifier_model_matrix.py`.
- N16R4 CPU/cache-only launcher:
  `scripts/download_c3_coarse_classifier_model_zoo_n16r4.sh`.
- Focused tests:
  `tests/test_c3_coarse_classifier_model_matrix.py`.

The launcher clears `CUDA_VISIBLE_DEVICES` and only downloads/caches model
weights; it does not start training or evaluation.

## Local Verification

Commands:

```powershell
C:\Users\skywalker\.conda\envs\torch_1\python.exe -m py_compile tools\bata\c3_coarse_classifier_model_matrix.py tests\test_c3_coarse_classifier_model_matrix.py
bash -n scripts/download_c3_coarse_classifier_model_zoo_n16r4.sh scripts/run_c3_tcn_coarse_probe_gpu1_20260701.sh
C:\Users\skywalker\.conda\envs\torch_1\python.exe -m pytest tests\test_c3_coarse_classifier_model_matrix.py -q
C:\Users\skywalker\.conda\envs\torch_1\python.exe tools\bata\c3_coarse_classifier_model_matrix.py --download --dry-run --tier first_wave --output-json logs\c3_model_matrix_dry_run_local.json
git diff --check
```

Results:

- py_compile: pass;
- launcher `bash -n`: pass;
- focused pytest: `4 passed`;
- matrix dry-run JSON: pass;
- diff check: pass.

## Next Actions

1. Commit and push this C3 coarse classifier model-matrix update.
2. Sync the remote clean clone.
3. Run remote static precheck.
4. Start CPU-only first-wave model download on the N16R4 login node under
   `/data/run01/sczc063/yuzibo/model_zoo_cache/c3_coarse_classifier`.
5. After `c3_oracle_full_g1` releases GPU1, run the existing temporal-TCN wave,
   then add the broader downloaded model families into the same
   `p_action`/`|delta p_action|` indirect-selection benchmark.
