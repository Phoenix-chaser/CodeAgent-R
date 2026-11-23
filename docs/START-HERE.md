# 上手指南

[中文首页](../README.md) · [English overview](../README_EN.md) · [实验成绩单](EXPERIMENT-RESULTS.md)

## 运行第一个示例

使用 Python 3.10+，并确认当前 Python 已安装 `venv` 模块。完整执行测试依赖 Bash、Git 和 POSIX 进程工具，适合在 Linux 或 WSL2 上运行。第一次使用可克隆公开仓库并创建独立环境。

```bash
git clone https://github.com/Phoenix-chaser/CodeAgent-R.git
cd CodeAgent-R
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m agentlab doctor
python examples/p01_minimal.py
```

`doctor` 列出实际探测到的工具和依赖，其中 `sandbox.*` 几行说明判分实际使用的隔离。示例将一个带判分信息的私有任务投影成公开提示，最后打印 `canary 在 prompt 中: False`。这一步只用本地 CPU，基础包安装完毕即可离线执行。

## 验证开发环境

```bash
python -m pip install -e ".[dev,lint]"
python -m pytest tests -q
ruff check .
python scripts/export_tasks_v2.py --list-pools
```

测试不需要 GPU、模型权重或网络。最后一条命令列出 `v2 v2b v2c v2d`。这四批扩展任务与 `pilot` 合计 104 题、52 个来源组。要导出并核对 64/20/20 的划分，使用手册中的显式比例。[GPU 实验手册](POSTTRAIN-GPU.md#冻结任务池)

## 继续阅读

| 目的 | 从这里开始 |
| --- | --- |
| 看懂三个 seed 的收益，以及 dev 与 test 为什么会分歧 | [实验成绩单](EXPERIMENT-RESULTS.md) |
| 从本地任务池开始，按阶段采集、训练、选型、终评 | [GPU 实验手册](POSTTRAIN-GPU.md) |
| 查三轮 DPO 的 reference、β 和 dev 结果 | [DPO 实验记录](EXPERIMENT-DPO.md) · [按组自适应 β 笔记](ADAPTIVE-BETA.md) |
| 查判分漏洞、执行隔离和统计口径的复审与修复 | [复审记录 2026-11-23](REVIEW-2026-11-23.md) |
| 把实验结果发布成可从仓库复核的证据 | [GPU 实验手册 · 导出证据](POSTTRAIN-GPU.md#导出可公开的证据) |
| 查早期运行时、预算及数据契约的修正 | [历史复审记录](REVIEW-FIXES.md) |

## 分享实验结果

公开文档可使用任务数、通过率、模型版本与汇总图。教师凭据、私有测试、候选代码、逐题日志和 checkpoint 留在本地。`doctor` 输出及部分 manifest 会包含机器路径或服务地址，分享前请检查对应字段。

要公开逐题结果，使用 `scripts/export_evidence.py`。它只导出判定与计数，自动去掉模型输出、私有测试细节、服务地址、绝对路径和硬件型号。

从仓库根目录运行 `git diff --cached --stat` 查看暂存范围，再运行 `python scripts/privacy_check.py` 检查跟踪文件中的已知模式。[仓库忽略规则](../.gitignore)
