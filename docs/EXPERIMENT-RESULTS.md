# 代码修复实验成绩单

[项目首页](../README.md) · [GPU 实验手册](POSTTRAIN-GPU.md) · [DPO 实验记录](EXPERIMENT-DPO.md)

基座为 **Qwen2.5-Coder-7B-Instruct**。104 道 Python 修复题按 52 个来源组冻结为 train 64、dev 20、test 20。模型选择看 dev 执行通过率，终评逐题配对比较 Base 和选中模型。下列数字取自项目已有实验记录。

## 方法对照

| 方法 | dev 通过率 | test 通过率 | 实验条件 |
| --- | ---: | ---: | --- |
| Base | 50% | 60% | Greedy 基线 |
| 教师蒸馏 SFT，seed 42 | 60% | **65%** | dev 选中 epoch 3 |
| DPO 第 1 轮 | 55% | — | SFT reference，统一 β |
| DPO 第 2 轮 | 55% | — | SFT reference，按采样组调整 β |
| DPO 第 3 轮 | 50% | — | Base reference，按采样组调整 β |
| RFT 第 1 轮 | 70% | **65%** | 从 SFT adapter 继续训练 |
| RFT 第 2 轮 | 80% | 55% | 从第 1 轮 adapter 继续训练 |
| Hybrid | — | **65%** | 混合 202 条教师样本和约 150 条 RFT 样本 |
| Best-of-N | — | **65%** | 先 greedy，必要时最多追加 4 次采样，以公开测试筛选 |

DPO 三轮均止于 dev。Hybrid 和 Best-of-N 的记录只给出 test 成绩。Best-of-N 允许多次生成，所以它使用的推理预算高于 greedy。[方法对比图](images/method-comparison.png)

## 三个训练种子

| 训练 seed | dev 选型 | Base · test | 选中模型 · test | 差值 |
| ---: | --- | ---: | ---: | ---: |
| 42 | epoch 3 | 12/20 | **13/20** | +5 pp |
| 43 | 保留 Base | 12/20 | 12/20 | 0 pp |
| 44 | epoch 2 | 12/20 | 12/20 | 0 pp |

三次选择后的 test 分数平均比 Base 高 **1.7 个百分点**，三个差值的标准差为 2.4 个百分点。95% bootstrap 区间为 **[0, +5] 个百分点**。汇总脚本先对同一道题取跨 seed 的平均差值，再对 20 道题重采样。三个 seed 共用一套 test 题。[汇总实现](../scripts/aggregate_seeds.py)

seed 42 的 dev 从 10/20 升到 12/20，其中 3 题改善、1 题退化。test 从 12/20 升到 13/20，1 题改善、0 题退化。三个 seed 的整体 test 得分均达到或超过 Base。[三个 seed 的结果图](images/three-seed-results.png)

## 训练细节

| 训练设置 | 教师蒸馏 SFT | RFT 第 1 轮 | RFT 第 2 轮 |
| --- | --- | --- | --- |
| 起点 | Base | SFT adapter | RFT 第 1 轮 adapter |
| 训练数据 | 64 题，202 条去重、重放通过的教师解 | 约 50 题，约 150 条自生成样本 | 58 题，180 条自生成样本 |
| 学习率 / epoch | 2e-5 / 3 | 1e-5 / 2 | 5e-6 / 2 |
| dev → test | 12/20 → 13/20 | 14/20 → 13/20 | 16/20 → 11/20 |

SFT 使用 NF4 双量化 QLoRA，LoRA 秩为 16、α 为 32。训练记录为 78 步，峰值显存约 10.3 GB。候选指纹迁移将记录的采集开销降低了 56%。变异审计中，104 题里有 75 题的测试捕获了本轮生成的全部变异体。[候选迁移](../scripts/migrate_candidates.py) · [变异审计](../scripts/audit_mutations.py)

RFT 每题采样 8 次，保留经判分通过的修复，并尝试用诊断提示恢复失败样本。第 2 轮的采样通过率约为 78%，dev 又多对了两题，test 却比第 1 轮少对两题。已评测方法在这组 test 上的最高观察值为 13/20，另有 7 题在这些终评中始终未通过。[RFT 数据生成](../scripts/generate_rft_data.py)

## 早期实验如何演进

早期轮次使用过不同的任务集与评测协议。一份 14 题的记录从 7/14 变为 6/14，这些题曾同时用于采集和评测，采样也未固定 seed，只能用于流程诊断。下表保留试验沿革，正式对照使用上面的 20 题冻结 test。

| 轮次 | 数据或方法变化 | 当轮观察 |
| --- | --- | --- |
| 1 | 初版 SFT，71 条样本，dev/test 各 2 题 | 生成结果与 Base 逐字节相同 |
| 2 | 扩至 84 题，204 条样本，dev/test 各 12 题 | dev 多通过 1 题，test 少通过 1 题 |
| 3 | 修正标签，每题保留 1 个解，共 60 条 | 监督 token 降到上一轮的约 26%，该轮得分持平 |
| 4 | 扩至 104 题，加入变异审计和 3 个训练 seed | 三个 seed 的平均 test 差值为 +1.7 pp |
| 5–7 | 三轮 DPO，共用 50 对偏好样本 | dev 分别为 11、11、10 题，SFT 为 12 题 |
| 8–9 | 两轮 RFT，自生成训练样本增至约 150、180 条 | 第 2 轮的 dev 和 test 走势相反 |

## 复现与延伸阅读

复现时在本地保存 split manifest、候选及重放记录、训练 manifest、逐 epoch 权重哈希、dev 选择与配对终评报告。公开交流使用汇总统计，逐题生成、隐藏测试与原始日志留在本地。[GPU 实验手册](POSTTRAIN-GPU.md) · [分享实验结果](START-HERE.md#分享实验结果)

| 主题 | 参考工作 |
| --- | --- |
| 代码修复与执行反馈 | [SWE-Gym](https://arxiv.org/abs/2412.21139) · [FailForge](https://arxiv.org/abs/2608.08570) · [PerfCodeGen](https://arxiv.org/abs/2412.03578) · [Self-Execution Simulation](https://arxiv.org/abs/2604.03253) |
| 自生成数据与拒绝采样 | [GiFT](https://arxiv.org/abs/2502.11466) · [RFT](https://arxiv.org/abs/2308.01825) |
| 偏好优化与组内反馈 | [DPO](https://arxiv.org/abs/2305.18290) · [FEST](https://arxiv.org/abs/2605.15012) · [GRPO](https://arxiv.org/abs/2402.03300) |
