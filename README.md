# CodeAgent-R

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-QLoRA_NF4-red)
![transformers](https://img.shields.io/badge/transformers-HuggingFace-yellow)
![tests](https://img.shields.io/badge/tests-468_passed-brightgreen)
![license](https://img.shields.io/badge/license-MIT-green)
![status](https://img.shields.io/badge/experiment-complete-brightgreen)

**中文** | [English](README_EN.md)

面向代码修复的后训练实验平台：公开上下文构造 → 教师候选采集 → 测试验证 → 数据隔离 → QLoRA SFT → 基座与 adapter 配对评测。已有真实 Hugging Face / PEFT 单卡 SFT 与评测闭环；DPO 和 GRPO 仍是离线契约与 toy 验证。

## 实验结果（v6，2026-11-07）

| 集合 | Base | SFT | 变化 | 改善/退化 |
|---|---|---|---|---|
| dev（20 题，选型用） | 10/20 (50%) | 12/20 (60%) | +10.0 pp，逐 epoch 单调 | +3 / −1 |
| **test（20 题，冻结无偏，seed 42）** | **12/20 (60%)** | **13/20 (65%)** | **+5.0 pp** | **+1 / 0** |
| **test 3-seed 汇总** | 60% | — | **均值 +1.7pp，CI [0, +5]** | **0 退化** |

104 题 / 52 来源组基准（变异测试审计数据质量）、202 条重放验证样本、QLoRA 3 epoch、按 dev 执行 pass@1 选 checkpoint（Base 参与对照）。**3 训练 seed 复现：均值 +1.7pp（0/+5/0），零退化；dev 无增益时选择机制自动保留基座**——完整数字、失败分析与诚实边界见 [v6 实验结果](docs/EXPERIMENT-V6-RESULTS.md)。

### 实验演进：每轮失败都有数据定位

| 轮次 | 训练样本 | dev/test | 现象 | 根因 |
|---|---|---|---|---|
| v3 | 71 | 2/2 题 | 输出与 Base 逐字节相同 | 数据不足，只学到风格 |
| v4 | 204 | 12/12 题 | dev +1 / test −1 | 12 题噪声主导；隐藏测试不完备 |
| v5 | 60 | 12/12 题 | 无变化 | 一题一解把监督 token 砍到 26% |
| **v6** | **202** | **20/20 题** | **3 seed：+5/0/0pp，均值 +1.7，零退化** | 规模×3.6 + 质量门槛 + 评测分辨率×2 |

## 验证边界与能力

**v6 摘要：在 20 个未见测试任务上 pass@1 60% → 65%（+5 个百分点，+1 改善 / 0 退化，配对 bootstrap 95% 区间 [+0.00, +15.00]）。** 属方向一致的小幅提升，尚未达统计显著；dev 选择集上 50% → 60% 且逐 epoch 单调（50→50→55→60），改善全部来自隐藏边界测试的语义修复。此前轮次（71/204/60 样本）的失败根因已逐一定位：数据规模不足导致"学风格不学语义"、隐藏测试不完备导致同题样本矛盾、12 题评测集噪声主导。完整数字、失败分析与诚实边界见 [v6 实验结果](docs/EXPERIMENT-V6-RESULTS.md)。

v6 关键配置：104 题 / 52 来源组按 0.2/0.2 冻结为 64/20/20；教师候选按任务指纹跨实验迁移复用（省 56% 采集成本）；变异测试审计数据质量（75 题零幸存变异体）；202 条重放验证样本 / 22,706 监督 token；QLoRA 3 epoch 78 步；每 epoch 不可变快照，按 dev 执行 pass@1 选择（Base 参与对照，平局保留 Base）；终评含训练与选择数据双重污染检查。

| 能力 | 已有实现 | 验证边界 |
|---|---|---|
| 数据与评测 | 104 题 / 52 来源组、变异测试审计、按来源组划分、去重/重放、三层 grader、候选指纹迁移 | 单文件平铺 Python 修复；真实仓库谱系待扩 |
| 单卡 SFT | NF4 QLoRA、完整模板与 assistant-only mask、监督 token 累积、每 epoch 保存、dev 执行选择 | 单 seed 单采样；多 seed 复现进行中 |
| 对照报告 | 默认 greedy、固定协议/预算校验、完整生成与判分诊断、配对区间、选择数据污染检查 | 20 题区间仍宽 |
| 运行时 | 工作区、快照、预算账本、同失败状态分支 | 进程控制不等于容器安全隔离；完整模型工具 episode 待验收 |
| DPO | 真实 trainer + 偏好构造 + 协议内选型（v7 实测：dev 55% 未超 SFT，按协议不上线；根因=教师对模型失败构成"已满足偏好"，梯度趋零，见 [v7 DPO 实验](docs/EXPERIMENT-V7-DPO.md)） | 未做 hard-negative 配对与多轮迭代 |
| GRPO | 共同状态契约、GRPO 数学检查 | 不是在线 RL |

## 快速开始

执行与评分使用 Bash/POSIX 工具，建议 Linux 或 WSL2。训练建议 Python 3.10–3.12；原生 Windows 后端未验收。

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q
python -m agentlab doctor
python scripts/prepare_data.py --help
python scripts/train_sft.py --help
python scripts/evaluate.py --help
python scripts/compare.py --help
```

GPU 环境先安装与驱动相容的 CUDA PyTorch，再安装训练和 API 依赖：

```bash
python -m pip install -e ".[dev,train,api]"
python -m pip check
```

`train` extra 给出候选兼容范围，尚未在目标 GPU 环境验证；实际实验需保存 `pip freeze`、CUDA、驱动和模型 revision。`agentlab` CLI 当前只有 `doctor`；训练等入口位于 `scripts/`，不是 `agentlab train-sft`。

## 复现与后续

完整流程（任务池导出与审计 → 按来源组冻结划分 → 教师采集与重放验证 → 训练 → dev 执行选型 → test 终评）见 [操作入口](docs/START-HERE.md) 与 [GPU 后训练实验手册](docs/POSTTRAIN-GPU.md)；一键脚本 `scripts/run_experiment_v2.sh` 支持阶段化断点续跑。教师候选可按任务指纹跨实验迁移复用（`scripts/migrate_candidates.py`，v6 省 56% 采集成本）。

后续方向：hard-negative 偏好配对（v7 DPO 负结果教训）、每题多次采样收窄区间（`scripts/aggregate_seeds.py` 已就绪）。**500+ 独立训练任务是建设目标；多次采样与改名不算独立任务，同源变体必须同 split。**

## 工程与研究内容

```text
新任务池 → source_group 划分与内容哈希 → train / dev / test
                 ↓ 仅 train / dev
公开指令 + 主文件 + 依赖文件 + 公开测试
                 ↓
教师候选 → 独立工作区三层判分 → 验证过的 SFT 数据
                 ↓
真实 tokenizer / QLoRA → dev 选择 adapter → 独立 test 配对报告
```

训练默认关闭 packing，避免跨样本注意力泄漏。监督目标来自完整 chat template，前缀和 padding 不进入 loss。梯度累积以有效监督 token 为分母，并处理不足一个累积窗口的尾批。保存数据、配置、模板、实现与模型版本信息，方便定位训练和推理差异。

研究问题应采用受控实验回答：等监督 token 下高质量数据是否有效；公开上下文是否改善跨文件修复；成功答案与失败恢复轨迹有何差异；同状态 DPO 是否带来额外收益。后两项需要继续实现真实轨迹采集与 DPO 训练，不能把现有 toy 数学检查写成已完成实验。

## 项目结构

```text
scripts/           数据准备、候选采集、真实 SFT、评测、对比入口
src/agentlab/
  benchmark/       任务、公开协议、三层 grader
  training/        训练数据与离线验证组件
  runtime/         工作区、执行、预算、快照
  trajectories/    同状态分支与偏好构造
  dpo/  rl/        离线契约、损失与数学验证
  experiments/    实验合同与报表
  providers/       OpenAI 兼容 provider
docs/              操作手册、实验结果与状态记录
tests/             不依赖真实模型/API 的自动化验证
```

[状态文件](docs/project-status.yaml) 记录当前范围与各阶段验收证据。

## 已知边界

- 评分会执行模型代码；当前工作区以用户权限运行，生产使用需补容器、网络及资源隔离。
- 完整 Agent 工具交互与在线 RL 尚待实现或验收。
- 内置任务不足以衡量真实仓库修复能力；外部任务的许可证、独立性和隐藏测试质量需要审计。
- `data/`、`results/`、`run-logs/`、`checkpoints/` 默认不提交；对外展示需另外整理不含密钥和私有材料的复现产物。

项目结构受 [pi](https://github.com/earendil-works/pi) 的极简 harness 思路启发，独立实现。License: MIT。
