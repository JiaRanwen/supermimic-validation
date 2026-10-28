# RLinf SuperMimic 工程验收

[English](README.md) | **简体中文**

本仓库为 [RLinf 中的 SuperMimic 集成](https://github.com/JiaRanwen/RLinf/tree/feat/supermimic) 提供一套**独立的工程验收工具**。仓库包含独立验收脚本、示例配置，以及（发布后）经过脱敏的测试证据。验收脚本有意放在 **RLinf 源码仓库之外**，不作为 RLinf 正式代码的一部分。

验收覆盖环境安装与导入、配置校验和单元测试、MuJoCo Warp 真实训练、检查点保存与恢复、独立运动跟踪和速度跟踪评测、GPU 性能测试、双卡 NCCL 同步、多 GPU 恢复及故障处理、短训练预算下的数值诊断。**通过这些测试不代表策略已收敛，也不代表复现了 SuperMimic 论文中的性能结果。**

## 仓库内容

```text
supermimic-validation/
├── README.md
├── README-zh.md
├── supermimic_acceptance.py
├── supermimic_acceptance.example.yaml
└── results/                         # 可选：公开的脱敏验收证据
```

验收脚本启动时仅依赖 Python 3 标准库。使用 `install.mode: fresh` 时，阶段 01 调用 RLinf 的 `requirements/install.sh` 安装实际训练依赖。该安装脚本可能生成或修改 RLinf 工作区中的依赖锁定或元数据文件；建议在可丢弃的 RLinf 检出副本中测试全新安装，结束后执行 `git status --short` 检查。

## 1. 目录布局（重要）

请把本仓库克隆到 **RLinf 仓库旁边，而非其内部**。动作数据和生成的测试结果也应保存在两个 Git 仓库之外：

```text
workspace/
├── RLinf/                          # 带有 SuperMimic 的功能分支
├── supermimic-validation/          # 本仓库
│   ├── supermimic_acceptance.py
│   ├── supermimic_acceptance.example.yaml
│   └── supermimic_acceptance.yaml  # 本地配置副本，请勿提交
├── data/lafan1_g1_40/              # 已重定向至 G1 的动作片段，不随仓库分发
├── supermimic_envs/                # 自动生成的虚拟环境
└── acceptance_results/            # 自动生成的报告与日志
```

**YAML 中的相对路径始终以 `supermimic_acceptance.py` 所在目录为基准解析，而非以终端当前工作目录为基准。** 在上述并列仓库布局中，应设置 `paths.repo: ../RLinf`、`data.lafan1: ../data/lafan1_g1_40`，并把环境和结果分别放在 `../supermimic_envs`、`../acceptance_results`。

如果采用其他目录布局，必须同步修改对应路径。例如，原开发机把 `supermimic_acceptance.py` 直接放在 `/mnt/public2/jiaranwen/`，而 `RLinf/`、`data/` 也是该目录的子目录，此时应使用：

```yaml
paths:
  repo: RLinf
  venv: supermimic_envs/{run_id}
  results: acceptance_results
  tools: .supermimic_tools

data:
  lafan1: data/lafan1_g1_40
  bones_seed: null
```

这只是**原开发机的目录示例**，并非公开示例 YAML 的默认布局；请根据实际位置选择。也可以用命令行 `--repo /absolute/path/to/RLinf` 单独覆盖仓库路径，但仍须正确设置数据、环境和结果路径。

## 2. 运行条件

- Linux 计算节点；涉及 GPU 的阶段需要 NVIDIA CUDA，以及 RLinf SuperMimic 分支支持的 MuJoCo Warp 运行环境。
- Python 3、Git、Bash、足够的磁盘空间；全新安装还需要访问依赖来源的网络连接。
- 单卡 Smoke 训练与评测至少需要**一张可用 GPU**；阶段 08、09 以及下面配置中的双卡阶段 12 需要同一节点有**两张可用 GPU**。
- 预先准备**重定向到 G1 的动作片段目录**（`clip_*.pt` 及 RLinf 动作库要求的元数据）。原始 LAFAN1 数据不能直接代替 G1 重定向结果；请参阅 RLinf 的 SuperMimic 使用指南。**本仓库不提供动作数据。**
- 具备创建虚拟环境和临时文件的权限。设置 `install.no_root: true` 时，所需系统库必须已预装。

请在**计算或开发节点**执行 GPU 阶段，而不是集群登录节点。使用 Slurm 时，应在运行前根据实际集群修改 `scheduler`、`slurm` 配置。

## 3. 克隆与配置

```bash
mkdir -p ~/supermimic-workspace
cd ~/supermimic-workspace

git clone --branch feat/supermimic https://github.com/JiaRanwen/RLinf.git RLinf
git clone https://github.com/JiaRanwen/supermimic-validation.git

cd supermimic-validation
cp supermimic_acceptance.example.yaml supermimic_acceptance.yaml
```

修改本地的 **`supermimic_acceptance.yaml`**，填写实际的 G1 重定向数据路径，并按照硬件条件设置调度器和 GPU 选项。公开示例默认采用上一节的并列目录布局。若要复现某一次历史验收，可将 RLinf 检出固定到该次验收记录的 Commit；参见[验收证据与限制](#6-验收证据与限制)。**公开的验收脚本应与产生 PASS 结果时使用的版本一致**，包括数值比较容差等后续修复，不应上传未经修复的旧版本。

**缓存模式说明：**

```yaml
install:
  mode: fresh
  cache: isolated
  cache_dir: null
```

`isolated` 会自动在验收脚本目录下创建 `.supermimic_cache/<run_id>/`，因此此处 `cache_dir: null` **合法且符合预期**。相反，`cache: shared` **必须**指定非空的 `cache_dir`，如 `.supermimic_cache/shared`。`isolated` 表示各次运行缓存分离，**并非禁用缓存**。

全新安装建议使用唯一虚拟环境路径，如 `../supermimic_envs/{run_id}`。脚本会自动替换 `{run_id}`，且**不会删除已有虚拟环境**。

## 4. 运行验收

以下命令均在 `workspace/supermimic-validation/` 目录执行。

**仅预览计划，不实际执行：**

```bash
python3 supermimic_acceptance.py --mode full --dry-run
```

**仅检查全新安装：**

```bash
python3 supermimic_acceptance.py --mode full --stages 01 --verbose
```

此时阶段 00（预检查）和阶段 14（汇总）也会执行。即便只验收安装，阶段 00 仍要求 `data.lafan1` 指向有效数据目录；不要把阶段 00 的配置失败误判为安装器失败。

**单卡端到端 Smoke 测试：**

```bash
CUDA_VISIBLE_DEVICES=0 python3 supermimic_acceptance.py --mode smoke --verbose
```

**完整工程验收（两张可用 GPU）：**

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 supermimic_acceptance.py --mode full --verbose
```

示例配置刻意关闭了阶段 10/10b（论文规模）、11（多节点）、13（长训练）、13b（BONES-SEED）。已关闭或未选择的阶段标记为 `NOT_APPLICABLE`，**不是 `PASS`**。`full` 模式**不代表**已复现论文规模结果。

**复用现有环境，只测试指定阶段：**先将本地 YAML 的 `install.mode` 设为 `reuse`，并将 `paths.venv` 改为**真实存在的虚拟环境目录**，再运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 supermimic_acceptance.py \
  --mode full --stages 04,05,06,07,08,09,12 --verbose
```

这里包含阶段 04，是因为阶段 05/06/07 依赖**本次验收运行**在短训练中生成的检查点。单独指定 `--stages 06` 不会自动找到旧验收目录中的检查点。若所选阶段不包含 01，脚本要求 `install.mode: reuse`；保留 `fresh` 会被拒绝。

## 5. 输出及其含义

每次运行都会在 `../acceptance_results/` 下创建独立的运行目录：

```text
acceptance_results/<run_id>/
├── summary.md                  # 人类可读的结论及阶段状态
├── manifest.json               # 阶段元信息、配置、命令及诊断
├── logs/                       # 详细阶段日志
├── checkpoints/                # 测试生成的检查点
├── evaluations/                # 评测输出
├── metrics/                    # 性能与数值指标
└── env/                        # 安装环境信息（适用时）
```

阶段 14 成功仅表示**汇总报告已写出**，不能覆盖前面阶段的失败。请阅读 `summary.md` 或 `manifest.json` 判断最终结果及各阶段状态。即便 Profiling 子进程的退出码为 0，只要某一指定规模发生 CUDA OOM，工程验收阶段依然可能正确地判定为失败。

状态含义：`PASS` = 达到验收要求；`FAIL` = 执行了但未通过；`BLOCKED` = 先决条件或资源不足；`NOT_APPLICABLE` = 主动跳过、功能关闭或复用安装环境；`NOT_RUN` = 之前失败导致未执行。

在共享 GPU 服务器上，建议 Profiling 前执行 `nvidia-smi`。其他用户的显存占用也可能触发 CUDA OOM；如果因此降低了测试规模，应如实报告**实际规模**，不能声称完成了更大规模的测试。

## 6. 验收证据与限制

与 RLinf SuperMimic PR 对应的工程验收在共享开发服务器上使用**两张 NVIDIA A800 80 GB GPU**，验证的 RLinf 代码 Commit 为 **`a7dbdd95`**。随后 PR 分支更新至 **`b5573d2b`**，主要包含文档更新。

- 完整工程验收中的阶段 **02–09：PASS**。
- 阶段 **12：PASS**；一个短预算变体共训练 **12 次更新**，检查了数值有限性及计数器单调性。
- 最终完整工程验收使用 **`install.mode: reuse`**，因此其阶段 01 **不代表**全新安装验证。若在独立运行中完成全新安装，应另外提供该次运行的报告作为证据。
- 评测中的 `normalized_return` 仅约从 **0.001892 变化至 0.001893**，失败率维持在 **1.0**，**不足以证明有意义的策略性能改善**。
- **尚未验证**论文预算下的完整训练、长时间训练收敛或论文报告的最终性能指标。
- Profiling 的环境数量和显存数据应以**实际公布的验收记录**为准；不可从当前示例 YAML 或理论外推结果推断曾经完成的测试规模。

若在 `results/` 公布脱敏报告，应记录**准确的验收 Run ID**、RLinf Commit、安装模式（全新/复用）、GPU 配置、实际 Profiling 规模、被关闭的阶段，以及对应的验收脚本版本。

**发布文件前请检查：**`manifest.json`、日志和 YAML 是否包含机器本地路径、用户名、内部主机名/IP、密码、令牌、代理 URL 或受许可证约束的数据。不要上传虚拟环境、动作数据、训练检查点或未审核的原始日志；脱敏时应保留可审计的阶段状态和量化结果。

## 7. 相关 RLinf 文档

- [SuperMimic 英文使用指南（RST）](https://github.com/JiaRanwen/RLinf/blob/feat/supermimic/docs/source-en/rst_source/examples/embodied/supermimic.rst)
- [SuperMimic 中文使用指南（RST）](https://github.com/JiaRanwen/RLinf/blob/feat/supermimic/docs/source-zh/rst_source/examples/embodied/supermimic.rst)
- [SuperMimic 英文算法参考（RST）](https://github.com/JiaRanwen/RLinf/blob/feat/supermimic/docs/source-en/rst_source/reference/algorithms/supermimic.rst)
- [SuperMimic 中文算法参考（RST）](https://github.com/JiaRanwen/RLinf/blob/feat/supermimic/docs/source-zh/rst_source/reference/algorithms/supermimic.rst)
- [RLinf SuperMimic 集成分支](https://github.com/JiaRanwen/RLinf/tree/feat/supermimic)

本仓库提供的是**工程验收证据**，不是官方 Benchmark 排行榜，也不是对 SuperMimic 论文的独立性能复现。
