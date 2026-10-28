# SuperMimic Validation for RLinf

**English** | [简体中文](README-zh.md)

This repository provides an **external engineering acceptance suite** for the [SuperMimic integration in RLinf](https://github.com/JiaRanwen/RLinf/tree/feat/supermimic). It contains a standalone acceptance runner, a tracked default YAML configuration, and a summarized validation report. The acceptance runner is intentionally **not part of the RLinf source tree**.

The suite checks installation and imports, configuration and unit tests, real MuJoCo Warp training, checkpoint/restart, independent motion and velocity evaluations, GPU profiling, two-GPU NCCL synchronization, multi-GPU recovery/fault handling, and short-budget numerical diagnostics. Passing these checks **does not establish policy convergence or reproduction of the SuperMimic paper's results**.

## Repository contents

```text
supermimic-validation/
├── README.md
├── supermimic_acceptance.py
├── supermimic_acceptance.yaml
├── supermimic_validation_summary.json  # published, sanitized summary
└── results/                         # optional, further sanitized evidence
```

The runner requires only the Python 3 standard library **to launch**. Stage 01 installs the actual training dependencies with RLinf's `requirements/install.sh` when `install.mode: fresh` is selected. This installation script may generate or modify dependency lock/metadata files in the RLinf checkout; use a disposable checkout for a fresh-install test and inspect `git status --short` afterward.

## 1. Directory layout (important)

Clone this repository **next to**, not inside, the RLinf checkout. Keep motion data and generated results outside both Git repositories:

```text
workspace/
├── RLinf/                          # SuperMimic feature branch
├── supermimic-validation/          # this repository
│   ├── supermimic_acceptance.py
│   ├── supermimic_acceptance.yaml  # tracked default config
│   └── supermimic_acceptance.local.yaml  # optional, private override
├── data/lafan1_g1_40/              # prepared G1 clips; not distributed here
├── supermimic_envs/                # generated virtual environments
└── acceptance_results/            # generated reports and logs
```

**All relative paths in the YAML are resolved relative to the directory containing `supermimic_acceptance.py`**, not the shell's working directory. In this sibling layout, set `paths.repo: ../RLinf`, `data.lafan1: ../data/lafan1_g1_40`, and put environments/results under `../supermimic_envs` and `../acceptance_results`.

## 2. Prerequisites

- Linux compute host (the GPU stages require NVIDIA CUDA and the MuJoCo Warp-compatible runtime supported by the RLinf SuperMimic branch).
- Python 3, Git, Bash, sufficient disk space, and network access to install dependencies for a fresh-environment run.
- **One available GPU** for smoke training/evaluation; **two available GPUs on one node** for stages 08 and 09 and the two-GPU stage 12 configuration below.
- A prepared directory of **retargeted G1 motion clips** (`clip_*.pt`, with the motion-library metadata as required by RLinf). The original LAFAN1 dataset is not a drop-in replacement; follow the RLinf SuperMimic user guide's data-preparation instructions. Motion data is **not** included here.
- Permission to create a virtual environment and temporary files. With `install.no_root: true`, required system libraries must already be installed.

Run local GPU stages on a **compute/development node**, not on a cluster login node. For Slurm, configure the `scheduler` and `slurm` sections as appropriate before starting.

## 3. Clone and configure

```bash
mkdir -p ~/supermimic-workspace
cd ~/supermimic-workspace

git clone --branch feat/supermimic https://github.com/JiaRanwen/RLinf.git RLinf
git clone https://github.com/JiaRanwen/supermimic-validation.git

cd supermimic-validation
```

The tracked **`supermimic_acceptance.yaml`** already assumes the sibling directory structure above and is loaded automatically; no copying or renaming is necessary. Adjust the paths to your prepared retargeted data and the available GPUs as needed. Do **not** commit machine-specific path changes to this tracked file.

To keep the working tree clean, optionally create `supermimic_acceptance.local.yaml` (ignored by Git) and invoke `python3 supermimic_acceptance.py --config supermimic_acceptance.local.yaml ...` instead. The RLinf checkout may be pinned to a specific commit for reproduction; see [Evidence and limitations](#6-evidence-and-limitations).

**Publish the exact acceptance script revision used for the PASS results**, including any numerical-tolerance fixes, rather than an older unpatched copy.

**Cache modes:**

```yaml
install:
  mode: fresh
  cache: isolated
  cache_dir: null
```

`isolated` automatically creates `.supermimic_cache/<run_id>/` under the validation runner's directory, so `cache_dir: null` is **valid**. By contrast, `cache: shared` **requires** a non-null `cache_dir`, such as `.supermimic_cache/shared`. `isolated` means separate caches per run, **not** no caching.

For a fresh installation, use a unique virtual-environment path such as `../supermimic_envs/{run_id}`. The runner replaces `{run_id}` automatically and **does not delete existing environments**.

## 4. Run the acceptance suite

Run all commands from `workspace/supermimic-validation/`.

**Preview without executing:**

```bash
python3 supermimic_acceptance.py --mode full --dry-run
```

**Fresh-install check only:**

```bash
python3 supermimic_acceptance.py --mode full --stages 01 --verbose
```

Stage 00 (preflight) and stage 14 (summary) also execute. Even for an install-only run, stage 00 requires a valid `data.lafan1` directory. Do not interpret a stage 00 configuration failure as an installer failure.

**End-to-end smoke test (single GPU):**

```bash
CUDA_VISIBLE_DEVICES=0 python3 supermimic_acceptance.py --mode smoke --verbose
```

**Full configured engineering matrix (two available GPUs):**

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 supermimic_acceptance.py --mode full --verbose
```

With the published default configuration, stages 10/10b (paper-scale), 11 (multi-node), 13 (long training), and 13b (BONES-SEED) are intentionally disabled. A disabled or unselected stage is marked `NOT_APPLICABLE`, **not `PASS`**. Full mode does **not** imply paper-scale reproduction.

**Run selected stages using an already installed environment:** change your configuration to `install.mode: reuse` and set `paths.venv` to the **existing** virtual environment before running:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 supermimic_acceptance.py \
  --mode full --stages 04,05,06,07,08,09,12 --verbose
```

Stage 04 is included because stages 05/06/07 depend on the short-training checkpoint created **within the same acceptance run**. `--stages 06` alone does not automatically find checkpoints from older result directories. Selecting stages without 01 while keeping `install.mode: fresh` is rejected by the runner.

## 5. Outputs and interpretation

Each execution creates a new run directory under `../acceptance_results/`:

```text
acceptance_results/<run_id>/
├── summary.md                  # human-readable conclusions and stage statuses
├── manifest.json               # stage metadata, configuration, commands, diagnostics
├── logs/                       # detailed stage logs
├── checkpoints/                # generated test checkpoints
├── evaluations/                # generated evaluation results
├── metrics/                    # profiling and numerical results
└── env/                        # environment information, when installed
```

A successful stage 14 means the **report was written**; it does not override an earlier stage failure. Read `summary.md` or `manifest.json` for the overall result and stage statuses. A profiling subprocess may exit successfully while the acceptance stage correctly fails because one tested environment size encountered CUDA OOM.

Typical status meanings: `PASS` = criteria passed; `FAIL` = an attempted check failed; `BLOCKED` = unmet prerequisites/resources; `NOT_APPLICABLE` = intentionally excluded, disabled, or installation reused; `NOT_RUN` = not executed after an earlier failure.

On shared GPU hosts, check `nvidia-smi` before profiling. A `CUDA out of memory` can be caused by other users' GPU allocations; any reduced profiling sizes must be reported explicitly rather than described as tests of a larger configuration.

## 6. Evidence and limitations

The engineering acceptance reported for the RLinf SuperMimic PR was performed on a shared development server with **two NVIDIA A800 80 GB GPUs**, against RLinf commit **`a7dbdd95`**. The PR branch was subsequently updated to **`b5573d2b`** for documentation changes.

- Engineering stages **02–09: PASS** in the reported full-mode run.
- Stage **12: PASS** for one short-budget variant, **12 updates**; finite numerical metrics and monotonic counters were checked.
- That final full-mode run used **`install.mode: reuse`**. Its stage 01 was **not** a fresh-install validation; fresh installation, where performed in a separate run, must be cited separately with its own report.
- The observed evaluation return changed only about **0.001892 → 0.001893**, while **failure rate remained 1.0**. These observations **do not demonstrate meaningful policy improvement**.
- **Paper-budget training, long-run convergence, and reproduction of the SuperMimic paper's reported performance were not verified.**
- Profiling environment sizes and memory figures should be taken from the **actual published metrics for that run**; do not infer them from the current default YAML or from theoretical extrapolations.

The repository-root `supermimic_validation_summary.json` is an abbreviated public result. If further sanitized reports are published in `results/`, they should name the **exact validation-run ID**, the tested RLinf commit, whether environment installation was fresh or reused, GPU configuration, evaluated profiling sizes, and any disabled stages. It should reference the acceptance script revision used for the run.

**Before publishing artifacts:** review `manifest.json`, logs, and YAML for machine-specific paths, usernames, internal hostnames/IPs, credentials, access tokens, proxy URLs, and dataset/licensing restrictions. Do not publish virtual environments, motion data, generated checkpoints, or unreviewed raw logs. Retain stage statuses and quantitative results when sanitizing so that the evidence remains auditable.

## 7. Related RLinf documentation

- [SuperMimic user guide (English RST)](https://github.com/JiaRanwen/RLinf/blob/feat/supermimic/docs/source-en/rst_source/examples/embodied/supermimic.rst)
- [SuperMimic algorithm reference (English RST)](https://github.com/JiaRanwen/RLinf/blob/feat/supermimic/docs/source-en/rst_source/reference/algorithms/supermimic.rst)
- [RLinf SuperMimic integration branch](https://github.com/JiaRanwen/RLinf/tree/feat/supermimic)

This repository is **engineering validation evidence**, not an official benchmark leaderboard or an independent reproduction of the SuperMimic paper.