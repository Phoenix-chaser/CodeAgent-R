# v6 实验最终结果（2026-11-07）

首个方向一致的正结果。此前轮次（v3/v4/v5）的失败模式已在 v6 中逐项修复：
数据规模（22,706 监督 token，为 v3 的 3.6 倍）、任务多样性（104 题 / 52 来源组 /
40+ 缺陷家族）、数据质量门槛（变异测试审计）、评测分辨率（dev/test 各 20 题 ×
10 来源组）、checkpoint 选择（dev 执行 pass@1，含未训练 Base 对照）。

## 配置

| 项 | 值 |
|---|---|
| 基座 | Qwen/Qwen2.5-Coder-7B-Instruct @ c03e6d35 |
| 训练 | QLoRA NF4 双量化、r16/α32、lr 2e-5、3 epoch、78 步、峰值 10.3GB、236s |
| 数据 | OpenAI 兼容教师 API；train 64 题 202 条去重样本（重放验证全过）；旧候选按 task_hash 迁移复用 472 条，新增付费采集 376 槽 |
| 数据质量 | 全部任务通过变异测试审计（104 题中 75 题零幸存变异体，其余为已确认等价变异） |
| 选择 | 每 epoch 保存不可变快照，按 dev 执行 pass@1 选择，平局保留 Base → 选中 epoch-3 |

## 结果

| 集合 | Base | SFT | 变化 | 改善/退化 |
|---|---|---|---|---|
| dev（20 题，用于选型） | 10/20 (50%) | 12/20 (60%) | +10.0 pp | +3 / −1 |
| **test（20 题，冻结无偏）** | **12/20 (60%)** | **13/20 (65%)** | **+5.0 pp** | **+1 / 0** |

- test 配对 bootstrap 95% 区间：**[+0.00, +15.00] pp**（10000 次重采样，任务级配对）
- dev 逐 epoch pass@1：50% → 50% → 55% → 60%（单调上升）
- dev loss：0.432 → 0.261 → 0.230 → 0.225
- 污染检查：训练数据与 checkpoint 选择数据对 test 均 clean；协议哈希一致；
  零基础设施错误；greedy 解码、seed 42、2048 tokens、NF4 推理

## 失败分析（test 上 Base 与 SFT 共同未通过的 7 题）

`pilot_option_pairs`、`pilot_clip_numeric`（公开测试失败）、`v2b_fill_defaults`、
`v2b_gcd_list`、`v2b_lcm_pair`、`v2b_safe_divide`、`v2b_severity_rank`（私有边界
失败）。其中 v2b 系多数来源组整体落在 test（组级隔离下训练从未见过该家族），
SFT 未修复属预期；`v2b_http_label`（来源组同样在 test）被 SFT 修复，是唯一的
跨家族语义迁移案例（边界大小写与未知码处理）。

## 多训练 seed 复现（2026-11-07 补充）

| seed | dev 选择决策 | test Base→SFT | Δ (pp) | 改善/退化 |
|---|---|---|---|---|
| 42 | epoch-3（dev 50→60%） | 60% → 65% | +5.0 | +1 / 0 |
| 43 | **保留 Base**（dev 无增益） | 60% → 60% | 0 | 0 / 0 |
| 44 | epoch-2（dev 50→55%） | 60% → 60% | 0 | 0 / 0 |

**三 seed 均值 +1.7pp ± 2.4（范围 0~+5），任务级 bootstrap CI [0, +5]，零退化。**
单 seed +5pp 含运气成分；dev 无增益时选择机制自动保留基座（seed 43 实证），
任何 seed 都未在 test 上退化——"dev 没收益就不上线"的防线跨 seed 成立。
dev 增益不完全迁移到 test（seed 44），20 题分辨率仍是天花板。

## 诚实的边界

- 多 seed 均值 +1.7pp，CI [0, +5]：方向为正但幅度小、未达显著
- 每任务单次 greedy 生成；任务为单文件平铺 Python L2/L3 修复
- 教师数据单模型、单温度策略

## 复现材料

任务池（104 题/5 批）、split-manifest、候选迁移与采集 manifest、
dataset-manifest（重放验证）、训练 manifest 与逐 epoch 快照哈希、
selection.json（dev 选型）、base/selected 的 test 逐题原始输出与对比报告。
Git 提交链覆盖全部代码演进；运行产物不入库（见 .gitignore）。
