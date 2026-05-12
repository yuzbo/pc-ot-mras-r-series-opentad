# N16R4 Adapter Quality Migration - 2026-05-12

## Scope

Migrated the THUMOS14 Adapter + ActionFormer detached quality-rescore experiment to the Beijing SCC Lingshui N16R4 platform under the required remote-only workspace:

```text
~/run/yuzibo
```

Remote code directory:

```text
~/run/yuzibo/OpenTAD_Back_check
```

Remote THUMOS14 directory:

```text
~/run/yuzibo/thumos14
```

## Data Mapping

Local OpenDataLab raw video source:

```text
E:\下载\THUMOS14_video\OpenDataLab___THUMOS14_video\raw
```

Split mapping:

```text
Validation Data/validation      -> ~/run/yuzibo/thumos14/train
Test Data/TH14_test_set_mp4     -> ~/run/yuzibo/thumos14/test
```

Expected counts:

```text
train: 1010 mp4
test: 1574 mp4
```

Annotation uploaded:

```text
~/run/yuzibo/thumos14/annotations/thumos_14_anno.json
~/run/yuzibo/thumos14/annotations/category_idx.txt
```

## Claude CLI Review

Review channel:

```text
claude.cmd -p --permission-mode plan --effort xhigh --output-format text
```

Reviewed files:

```text
OpenTAD_Back/configs/adatad/thumos/input_random_fixed_50pct_adapter_quality_rescore_detached_n16r4.py
OpenTAD_Back/scripts/check_n16r4_adapter_quality_workspace.sh
OpenTAD_Back/scripts/run_adapter_quality_rescore_n16r4.sbatch
OpenTAD_Back/scripts/setup_n16r4_opentad_env.sh
logs/sync_thumos_n16r4.ps1
```

Main concerns:

- N16R4 config inherited the parent `work_dir`, risking checkpoint/log collision.
- Slurm script lacked explicit time, memory, and CPU resources.
- Slurm log path was relative to submission CWD.
- `BASE_PORT` was fixed and could collide across jobs.
- Preflight only checked that train/test directories existed, not that data upload was complete.
- Pretrained checkpoint path was hard-coded in preflight instead of read from config.
- Environment script used floating `decord`, `timm`, and transitive `mmengine`.

Accepted fixes:

- Added N16R4-specific `work_dir`.
- Added `--cpus-per-task=8`, `--time=48:00:00`, and `--chdir` to sbatch.
- Did not set `--mem`, because N16R4 rejects explicit memory requests and assigns 55GB per GPU by policy.
- Derived `BASE_PORT` from `SLURM_JOB_ID` when available.
- Set N16R4 `NAME` to match the N16R4 `work_dir`.
- Tightened preflight to require exact THUMOS14 mp4 counts.
- Read the pretrained checkpoint path from the loaded config.
- Added pinned `mmengine==0.10.3`, `decord==0.6.0`, and `timm==0.6.13`.
- Added a successful-env sentinel for idempotent environment setup reruns.
- Added `/etc/profile` sourcing before `module load` because Slurm batch shells do not initialize `module` automatically on this platform; profile loading is wrapped with temporary `set +u` because the platform profile references unset variables.
- Split setup behavior so Slurm job can complete conda/PyTorch linking only (`SETUP_TORCH_ONLY=1`), while pip/OpenMMLab dependencies can be installed from the login node where network resolution is available.
- Switched the Slurm conda/PyTorch linking stage to `mamba install --offline`, using packages already cached in `~/run/yuzibo/conda_pkgs`, because compute nodes cannot resolve the configured conda mirror.

Deliberately not accepted:

- `torch.cuda.is_available()` is not asserted in environment setup because setup runs on a login node without guaranteed GPU allocation. The script asserts `torch.version.cuda == "11.8"` instead; Slurm logs `nvidia-smi -L` at job start.

## Verification

Local syntax checks:

```text
bash -n scripts/check_n16r4_adapter_quality_workspace.sh scripts/run_adapter_quality_rescore_n16r4.sbatch scripts/setup_n16r4_opentad_env.sh
python -m py_compile configs/adatad/thumos/input_random_fixed_50pct_adapter_quality_rescore_detached_n16r4.py
```

Remote sync verification showed the corrected markers in:

```text
work_dir = ..._n16r4
module load miniforge3/24.11
#SBATCH --cpus-per-task=8
BASE_PORT derived from SLURM_JOB_ID
decord==0.6.0
```

OpenTAD_Back commit:

```text
7b21d05 add n16r4 adapter quality launch scripts
```

## Current Status

As of the latest check on 2026-05-12:

- Video upload is still running from local Windows to N16R4.
- Upload is in `train` split and has not reached `test` split yet.
- Environment PyTorch-linking setup was resubmitted as Slurm job `979154` after login-node attempts were terminated during PyTorch linking, job `979139` exposed missing `module` initialization, job `979140` exposed `set -u` incompatibility with `/etc/profile`, and jobs `979142`/`979153` exposed missing DNS for PyPI/conda mirrors on compute nodes.
- Training has not been submitted yet.

Launch gates:

```text
1. train mp4 count == 1010
2. test mp4 count == 1574
3. setup_n16r4_opentad_env.sh completes and creates .opentad_n16r4_env_ok
4. bash scripts/check_n16r4_adapter_quality_workspace.sh passes
5. sbatch scripts/run_adapter_quality_rescore_n16r4.sbatch
```
