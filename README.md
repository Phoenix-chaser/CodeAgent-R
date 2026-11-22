# CodeAgent-R

**中文** · [English](README_EN.md)

给模型一段有缺陷的 Python 代码，让它提交修复，再运行测试，看它究竟多修好了几道题。CodeAgent-R 用这套流程比较 Qwen2.5-Coder-7B-Instruct 的监督微调（SFT）、偏好优化（DPO）、拒绝采样微调（RFT）和推理时采样。

[上手指南](docs/START-HERE.md) · [实验成绩单](docs/EXPERIMENT-RESULTS.md) · [GPU 实验手册](docs/POSTTRAIN-GPU.md) · [MIT 许可证](LICENSE)

## 冻结测试集上的表现

<picture>
  <source media="(max-width: 600px)" srcset="docs/images/three-seed-results-mobile.png">
  <img alt="在相同的 20 道测试题上，Base 通过 12 道。按 dev 选型后，三个训练种子分别通过 13、12、12 道。相对 Base 平均提升 1.7 个百分点。" src="docs/images/three-seed-results.png">
</picture>

在这 20 道题上，**多通过 1 题就是 5 个百分点**。seed 42 选中第 3 个 epoch，多修好 1 题。seed 43 在 dev 上保留 Base，seed 44 选中第 2 个 epoch，两者的 test 得分均与 Base 持平。按题重采样得到的 95% 区间为 **0 到 +5 个百分点**。

其他方法中，RFT 第 1 轮在 dev 上通过 14/20、test 上通过 13/20。第 2 轮的 dev 升到 16/20，test 却降到 11/20。三轮 DPO 的 dev 得分均低于 SFT，按协议止于 dev。[查看完整对照和各轮配置](docs/EXPERIMENT-RESULTS.md)

## 本地快速开始

在仓库根目录使用 Python 3.10+。安装基础包后，下面的示例可在 CPU 上运行，无需教师 API 密钥。

```bash
python -m pip install -e .
python -m agentlab doctor
python examples/p01_minimal.py
```

示例会打印模型可见的任务提示。末尾的 `canary 在 prompt 中: False` 表示用于检查泄漏的私有标记已从提示中剔除。运行完整测试套件见[上手指南](docs/START-HERE.md)。

## 一个修复如何被选中

| 阶段 | 做法 |
| --- | --- |
| 冻结任务 | 104 个单文件修复任务来自 52 个来源组。每组整体进入同一集合，报告使用 64 道 train、20 道 dev 和 20 道 test。 |
| 验证候选 | 模型读取指令、缺陷代码、必要依赖和公开测试。判分器另行运行隐藏测试，数据准备时重放候选修复。 |
| 训练与选择 | 用通过验证的答案做 NF4 QLoRA 微调，只监督目标 assistant token。每个 epoch 保存 adapter，和 Base 在相同 dev 题上比较执行通过率，平局保留 Base。 |
| 最终评测 | 锁定模型版本和解码预算，在冻结的 test 上逐题配对比较 Base 与选中模型，记录改善和退化。 |

## 代码导航

| 想看什么 | 入口 |
| --- | --- |
| 任务如何构造、判分和检查数据隔离 | [基准与判分器](src/agentlab/benchmark/) · [任务变异审计](scripts/audit_mutations.py) |
| 训练、checkpoint 选择和评测怎么衔接 | [训练实现](src/agentlab/training/) · [实验脚本](scripts/) · [GPU 实验手册](docs/POSTTRAIN-GPU.md) |
| 偏好训练为什么止于 dev | [三轮 DPO 实验](docs/EXPERIMENT-DPO.md) · [FEST 实现笔记](docs/FEST-ANALYSIS.md) |
| 工作区、执行预算和轨迹如何记录 | [运行时](src/agentlab/runtime/) · [同状态分支](src/agentlab/trajectories/) · [实验合同](contracts/) |

当前修复协议只替换一个 Python 主文件，依赖作为只读上下文。判分会执行候选代码，运行不受信任的候选需另配容器与资源隔离。完整工具交互和在线 RL 仍属后续工作。

本地保留教师凭据、原始生成、逐题日志和 checkpoint。公开报告使用汇总成绩与模型版本。[分享实验结果](docs/START-HERE.md#分享实验结果)

项目独立实现，运行时设计受 [pi](https://github.com/earendil-works/pi) 启发。
