# Repository Guidelines

## Project Structure & Module Organization
This workspace mixes code, experiments, and research notes. The main development target is [`OpenTAD_Back`](/E:/DeskTop/TAD/temrefuse-tad/OpenTAD_Back), with source under `opentad/`, training entrypoints in `tools/`, and model configs in `configs/`. Use `OpenTAD/` for comparison against earlier variants. Keep generated figures in `figures/`, monitoring scripts in `logs/`, and long-form records in `research-wiki/` or top-level `*.md` reports. Avoid committing large artifacts such as checkpoints or tarballs unless explicitly required.

## Build, Test, and Development Commands
Install Python dependencies from the target repo:
```powershell
cd OpenTAD_Back
pip install -r requirements.txt
```
Run training:
```powershell
python tools/train.py configs/adatad/thumos/input_random_fixed_50pct.py
```
Run evaluation:
```powershell
python tools/test.py <config> <checkpoint>
```
Useful local utilities in this workspace include:
```powershell
python figures/gen_fig_input_sampling_distributions_all.py --ann figures/cache/thumos_14_anno.json
powershell -ExecutionPolicy Bypass -File logs/monitor_current_servers_every2h.ps1
```

## Remote Servers
Use these SSH endpoints for the current THUMOS14 Adapter + ActionFormer experiments:
```powershell
ssh -p 35407 root@connect.cqa1.seetacloud.com
ssh -p 25876 root@connect.cqa1.seetacloud.com
```
Remote working directory:
```bash
/root/autodl-tmp/OpenTAD_Back_check
```

## Coding Style & Naming Conventions
Use 4-space indentation and follow existing Python style in `opentad/`. Prefer descriptive snake_case for functions, variables, config names, and experiment files such as `input_oracle_boundary_dense_tubelet2_50pct.py`. Keep config inheritance shallow and explicit. Add short comments only where control flow or tensor semantics are non-obvious.

## Testing Guidelines
There is no single unit-test suite; validation is mostly script-based. Before submitting changes, run at least one targeted smoke test:
```powershell
python tools/train.py <config> --id 0
```
For data or sampling changes, verify shapes, masks, and GT remapping on a small run. Store diagnostic plots in `figures/` and summarize key outcomes in `research-wiki/experiments/`.

## Claude Discussion & Review Protocol
Use Claude CLI as a recurring external reviewer for this research track. Before committing to a new model direction, launching long GPU runs, or changing the main Adapter + ActionFormer route, run a read-only Claude discussion about the current evidence, failure modes, candidate solutions, expected metric impact, and stop/continue gates. Repeat this discussion after major experiment results, especially when a run underperforms the random-fixed Adapter baseline or when the next route would consume a full GPU training cycle.

After every implementation is landed locally, request a Claude CLI code review before deployment or long training. The review must be read-only and should check implementation correctness, train/val/test contract alignment, GT leakage risk, tensor shape and mask handling, config consistency, and experiment validity. Apply accepted review fixes, rerun the relevant verification, and then create a git commit for the reviewed implementation before syncing or launching remote experiments. Do not include large artifacts, datasets, checkpoints, or unrelated dirty-worktree changes in that commit. Record the Claude discussion/review summary, reviewer concerns, accepted fixes, verification command output, and resulting commit hash in `research-wiki/experiments/` or the active top-level report.

Suggested commands:
```powershell
claude.cmd -p --permission-mode plan --effort xhigh --output-format text "<direction or solution discussion prompt>"
claude.cmd -p --permission-mode plan --effort xhigh --output-format text "<read-only code review prompt>"
```

## Commit & Pull Request Guidelines
Follow the observed git style in `OpenTAD_Back`: short imperative subjects, often with a scope or bug-fix prefix, for example `fix duplicate segment removal` or `upload videomae conversion script`. PRs should include: purpose, changed configs/modules, expected metric impact, exact run commands, and links to logs or figures. Include screenshots only for visualization changes.

## Security & Configuration Tips
Do not hardcode secrets. Paths in configs often point to `/root/autodl-tmp/...` on remote servers; keep local overrides separate. When syncing to servers, copy code and lightweight metadata only, not datasets or checkpoints.
