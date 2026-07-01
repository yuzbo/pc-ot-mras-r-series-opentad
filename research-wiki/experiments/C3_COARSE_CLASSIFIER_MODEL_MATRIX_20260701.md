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
- Unified fine-tuning/diagnostic adapter:
  `tools/bata/train_lowres_action_probe.py`.
  The new `--probe-model matrix-zoo` branch keeps the existing frame-logit
  contract `[B,T]`, so the same AP/AUC/F1 and `p_action`/`|delta p_action|`
  indirect-selection diagnostics are reused for all candidates.
- N16R4 CPU/cache-only launcher:
  `scripts/download_c3_coarse_classifier_model_zoo_n16r4.sh`.
- GPU1-only image-backbone fine-tuning launcher:
  `scripts/run_c3_matrix_zoo_image_backbone_probe_gpu1_20260701.sh`.
- GPU1-only short-clip video-model fine-tuning launcher:
  `scripts/run_c3_matrix_zoo_video_probe_gpu1_20260701.sh`.
- Focused tests:
  `tests/test_c3_coarse_classifier_model_matrix.py` and
  `tests/test_lowres_action_probe.py`.

The launcher clears `CUDA_VISIBLE_DEVICES` and only downloads/caches model
weights; it does not start training or evaluation.

The two fine-tuning launchers are C3-mainline/GPU1-only and fail closed unless
`CUDA_VISIBLE_DEVICES=1`. They are meant to run after the current
`c3_oracle_full_g1` child releases GPU1.

Current fine-tuning support:

- `timm` image backbones: per-frame ImageNet-pretrained visual features plus a
  small temporal head. This tests whether stronger visual recognition fixes the
  weak `p_action` curve.
- `torchvision`/`pytorchvideo` short-clip backbones: sliding clip logits are
  interpolated back to `[B,T]` so the same indirect-selection diagnostics can be
  computed.
- SlowFast and Hugging Face VideoMAE are explicitly fail-closed in this first
  adapter unless a dedicated two-pathway/Transformers wrapper is added. They can
  be downloaded as teacher or upper-bound candidates, but are not silently
  treated as supported fine-tuning models.

## Frame-Segmentation Reader Matrix

The temporal-TCN probe was expanded from small TCN variants into a broader
frame-segmentation reader matrix. These models are still diagnostic-only coarse
action/background classifiers. They are not detector heads and they do not
change AdaTAD, assignment, post-processing, or official evaluation.

Classic/low-cost variants:

- `lite`;
- `dilated`;
- `multiscale`;
- `motion`;
- `residual`;
- `gated`;
- `separable_dilated`;
- `causal_dilated`.

Stronger frame-segmentation inspired variants:

- `ms_tcnpp`: multi-stage residual/dilated TCN refinement, used as a stronger
  TCN-family baseline.
- `c2f_tcn`: coarse-to-fine temporal aggregation with a downsampled coarse path
  fused back into the dense frame axis.
- `asformer_lite`: local temporal convolution plus lightweight self-attention,
  inspired by action-segmentation Transformers such as ASFormer.
- `fact_lite`: frame-action cross-attention with two action/background tokens,
  inspired by FACT-style frame/action interaction.
- `temporal_mamba_lite`: gated long-kernel bidirectional scan, inspired by
  Mamba/SSM temporal modeling but implemented without an external Mamba
  dependency.

These are compact "inspired-lite" probes for fast screening under the existing
`[B,T]` frame-logit contract. They should be judged by whether their
`p_action(t)`, `|delta p_action|`, uncertainty, transition score, and sample
visualizations align better with true action intervals/boundaries than
MobileNet, CADF/TCN, or oracle-shell controls. They should not be described as
full ASFormer/FACT/Mamba reproductions.

## Local Verification

Commands:

```powershell
C:\Users\skywalker\.conda\envs\torch_1\python.exe -m py_compile tools\bata\train_lowres_action_probe.py tools\bata\c3_coarse_classifier_model_matrix.py tests\test_lowres_action_probe.py tests\test_c3_coarse_classifier_model_matrix.py
bash -n scripts/download_c3_coarse_classifier_model_zoo_n16r4.sh scripts/run_c3_tcn_coarse_probe_gpu1_20260701.sh scripts/run_c3_matrix_zoo_image_backbone_probe_gpu1_20260701.sh scripts/run_c3_matrix_zoo_video_probe_gpu1_20260701.sh
C:\Users\skywalker\.conda\envs\torch_1\python.exe -m pytest tests\test_lowres_action_probe.py tests\test_c3_coarse_classifier_model_matrix.py -q
C:\Users\skywalker\.conda\envs\torch_1\python.exe tools\bata\c3_coarse_classifier_model_matrix.py --download --dry-run --tier first_wave --output-json logs\c3_model_matrix_dry_run_local.json
C:\Users\skywalker\.conda\envs\torch_1\python.exe tools\bata\train_lowres_action_probe.py --help
git diff --check
```

Results:

- py_compile: pass;
- launcher `bash -n`: pass;
- focused pytest: `41 passed`;
- matrix dry-run JSON: pass;
- train-probe CLI help: pass;
- diff check: pass.

## Remote Static Verification

Remote clean clone:
`/data/run01/sczc063/yuzibo/OpenTAD_C3TCNCoarseProbe_20260701`.

GitHub branch:
`https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/c3-tcn-coarse-probe-20260701`.

Commit:
`49c793a` (`Add matrix-zoo coarse classifier probes`).

Commands/evidence:

```bash
git fetch origin codex/c3-tcn-coarse-probe-20260701
git reset --hard 49c793a
bash -n scripts/download_c3_coarse_classifier_model_zoo_n16r4.sh \
  scripts/run_c3_tcn_coarse_probe_gpu1_20260701.sh \
  scripts/run_c3_matrix_zoo_image_backbone_probe_gpu1_20260701.sh \
  scripts/run_c3_matrix_zoo_video_probe_gpu1_20260701.sh
/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python -m py_compile \
  tools/bata/train_lowres_action_probe.py \
  tools/bata/c3_coarse_classifier_model_matrix.py \
  tests/test_lowres_action_probe.py \
  tests/test_c3_coarse_classifier_model_matrix.py
/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python -m pytest \
  tests/test_lowres_action_probe.py \
  tests/test_c3_coarse_classifier_model_matrix.py -q
```

Result:

- remote HEAD: `49c793a`;
- launcher `bash -n`: pass;
- py_compile: pass;
- Linux focused pytest: `40 passed`;
- marker: `REMOTE_C3_MATRIX_ZOO_PRECHECK_PASS_49c793a`.

Frame-segmentation reader extension:

- commit: `6a0c7b3` (`Add frame segmentation coarse probe variants`);
- remote HEAD after sync: `6a0c7b3`;
- launcher `bash -n`: pass for TCN and matrix-zoo GPU1 launchers;
- py_compile: pass for `train_lowres_action_probe.py` and its focused tests;
- Linux focused pytest: `41 passed in 7.37s`;
- marker: `REMOTE_C3_FRAME_SEG_READER_PRECHECK_PASS_6a0c7b3`.

No GPU, Slurm child, training, evaluation, detector mAP, runtime/FLOPs, deploy,
paper claim, GPU0 fallback, or BH-SDC/DIVERGENT action occurred during this
static verification.

## Next Actions

1. Commit and push this C3 coarse classifier model-matrix/fine-tuning update.
2. Sync the remote clean clone.
3. Run remote static precheck for the new matrix-zoo fine-tuning branch.
4. Continue CPU-only first-wave model download on the N16R4 login node under
   `/data/run01/sczc063/yuzibo/model_zoo_cache/c3_coarse_classifier`.
5. After `c3_oracle_full_g1` releases GPU1, run the existing temporal-TCN wave,
   then run the broader downloaded model families through the same
   `p_action`/`|delta p_action|` indirect-selection benchmark.
