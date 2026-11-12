# CodeAgent-R

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-QLoRA_NF4-red)
![transformers](https://img.shields.io/badge/transformers-HuggingFace-yellow)
![tests](https://img.shields.io/badge/tests-470_passed-brightgreen)
![license](https://img.shields.io/badge/license-MIT-green)
![status](https://img.shields.io/badge/experiment-complete-brightgreen)

**中文** | [English](README_EN.md)

面向代码修复的后训练实验平台。对 Qwen2.5-Coder-7B 做 QLoRA SFT，在独立测试集上验证了正向收益，并实现了 DPO 偏好优化（负结果有根因分析）。

## 最终结果

| 集合 | Base | SFT | 变化 | 改善/退化 |
|---|---|---|---|---|
| dev（20 题，选型用） | 50% | 60% | +10 pp，逐 epoch 单调上升 | +3 / −1 |
| **test（20 题，冻结无偏）** | **60%** | **65%** | **+5 pp，0 退化** | **+1 / 0** |
| **test 3-seed 汇总** | 60% | — | **均值 +1.7 pp，CI [0, +5]** | **0 退化** |

完整数字与失败分析见 [实验结果](docs/EXPERIMENT-RESULTS.md)。

## 四轮实验演进：每轮失败都有数据定位

| 轮次 | 做了什么 | 训练样本 | 评测集 | 结果 | 根因分析 |
|---|---|---|---|---|---|
| 第 1 轮 | 初版 SFT | 71 | 2/2 题 | SFT 输出与 Base 逐字节相同 | 数据太少，只学到了输出风格而非语义 |
| 第 2 轮 | 扩任务池至 84 题 | 204 | 12/12 题 | dev +1 题 / test −1 题 | 评测集太小（噪声主导）；部分任务隐藏测试不完备导致训练标签矛盾 |
| 第 3 轮 | 修标签后"一题只留一解" | 60 | 12/12 题 | 无变化 | 砍掉了 74% 监督 token，数据饿死 |
| **第 4 轮** | **扩至 104 题 + 变异测试审计 + 3 seed** | **202** | **20/20 题** | **+1.7 pp，零退化** | 数据×2.8 + 质量门槛 + 评测分辨率×1.7，三项同时到位 |

**关键防线实证**：第 4 轮的 3 个训练 seed 中，有 1 个 seed 在 dev 上无增益——选择机制自动保留了 Base 而非强推 SFT。这意味着"dev 上没收益就不上线"的防退化设计跨 seed 成立。

## 核心方法

| 能力 | 实现 |
|---|---|
| 任务基准 | 104 题 / 52 来源组，变异测试审计数据质量（幸存变异体 = 可通过判分门槛的行为分歧假标签） |
| 数据隔离 | 按来源组冻结 train/dev/test（64/20/20），候选按任务指纹跨实验迁移（省 56% 采集成本） |
| SFT 训练 | NF4 QLoRA、完整 chat template 的 assistant-only mask、监督 token 加权梯度累积、每 epoch 不可变快照 |
| 模型选择 | 按 dev **执行通过率**（非 loss）选择 checkpoint，Base 参与对照，平局保留 Base |
| 评测 | 默认 greedy、固定协议哈希、配对 bootstrap 区间、训练与选择数据双重污染检查 |
| DPO | 真实 trainer + FEST 自适应 β（不可解任务 β₁=0.1 强信号 / 可解任务 β₂=0.01）+ hard-negative AST 聚类选择（见 [DPO 实验](docs/EXPERIMENT-DPO.md)、[FEST 分析](docs/FEST-ANALYSIS.md)） |

### DPO 负结果与根因

DPO 在 dev 上 55%（未超过 SFT 的 60%），按协议不上线。训练日志显示 `preference_accuracy=1.0` 全程不变——教师答案对模型自身失败构成"已满足偏好"，DPO 梯度趋零。基于 [FEST 论文](https://arxiv.org/abs/2605.15012)分析，改进方向是自适应 β + hard-negative 配对（已实现，待 GPU 验证）。

## 快速开始

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q          # 470 tests
python -m agentlab doctor
```

GPU 训练与评测：

```bash
python -m pip install -e ".[dev,train,api]"
python -m pip check
```

完整流程见 [操作入口](docs/START-HERE.md) 与 [GPU 实验手册](docs/POSTTRAIN-GPU.md)；一键脚本 `scripts/run_experiment_v2.sh` 支持阶段化断点续跑。

## 管线架构

```text
任务池（变异测试审计）→ 按来源组冻结划分 → train / dev / test
      ↓ 仅 train / dev
公开指令 + 缺陷代码 + 依赖 + 公开测试
      ↓
教师候选采集 → 三层判分 → 重放验证 → SFT 训练数据
      ↓
QLoRA SFT → 每 epoch 快照 → dev 执行选型 → test 终评
```

训练默认关闭 packing（避免跨样本注意力泄漏）。监督目标来自完整 chat template，前缀和 padding 不进入 loss。梯度累积以有效监督 token 为分母。

## 项目结构

```text
scripts/           数据准备、采集、训练、评测、对比、隐私审查入口
src/agentlab/
  benchmark/       任务、公开协议、三层 grader、变异测试审计
  training/        SFT / DPO 训练器、数据准备
  runtime/         工作区、执行、预算、快照
  trajectories/    同状态分支与偏好构造
  experiments/     实验合同与运行状态
  providers/       OpenAI 兼容 provider
docs/              实验结果、方法论文档
tests/             不依赖真实模型/API 的自动化验证
```

## 已知边界

- 评分会执行模型代码；生产使用需补容器与资源隔离。
- 完整 Agent 工具交互与在线 RL 尚未实现。
- 当前任务为单文件平铺 Python 修复；多文件修复需扩展输出协议。
- `data/`、`results/`、`checkpoints/` 不入库。

项目结构受 [pi](https://github.com/earendil-works/pi) 启发，独立实现。

## References

- [SWE-Gym](https://arxiv.org/abs/2412.21139) — execution-feedback agent training (ICML 2025)
- [FailForge](https://arxiv.org/abs/2608.08570) — failure recovery via diagnostic hints (2026)
- [FEST](https://arxiv.org/abs/2605.15012) — adaptive β / hard-negative selection for DPO
- [DPO](https://arxiv.org/abs/2305.18290) — Direct Preference Optimization (NeurIPS 2023)
- [RFT](https://arxiv.org/abs/2308.01825) — Rejection Sampling Fine-Tuning
- [GRPO](https://arxiv.org/abs/2402.03300) — Group Relative Policy Optimization

License: MIT.
