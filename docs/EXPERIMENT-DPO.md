# DPO 的三轮 dev 实验

[实验成绩单](EXPERIMENT-RESULTS.md) · [FEST 实现笔记](FEST-ANALYSIS.md)

DPO 从已选中的 SFT adapter 继续训练。参照线是同一批 20 道 dev 题上的 Base（10/20）和 SFT（12/20）。三轮 DPO 分别通过 11、11 和 10 题，均未超过 SFT，选型流程因此保留 SFT，三轮都没有 test 分数。

## 如何构造偏好对

在 64 道 train 题上，学生模型每题采样 6 次。384 次采样中有 207 次通过判分（约 54%）。将通过验证的教师答案作为 `chosen`，同一公开提示下学生未通过的答案作为 `rejected`，每题最多取一对，最终得到 50 对。采集耗时约 51 分钟。[偏好生成脚本](../scripts/generate_preferences.py)

训练时，两个答案共用相同的提示前缀，只对目标 assistant token 计算对数概率。reference 冻结，每个 epoch 保存 adapter。第 1 轮使用 β=0.1、学习率 5e-7，训练 1 个 epoch，共 50 步，耗时约 49 秒。[DPO 训练入口](../scripts/train_dpo.py) · [训练实现](../src/agentlab/training/hf_dpo.py)

## 改动与观察

| 轮次 | 冻结 reference | β | 训练记录 | dev |
| --- | --- | --- | --- | ---: |
| 1 | SFT adapter | 统一 0.1 | 偏好准确率约 1，reward margin 接近 0，loss 接近 log 2 | 11/20 |
| 2 | SFT adapter | 按采样组取 0.1 或 0.01 | 训练指标与第 1 轮相近 | 11/20 |
| 3 | Base | 按采样组取 0.1 或 0.01 | loss 从 0.72 降至 0.63 | 10/20 |

`reward_margin` 衡量 policy 相对 reference 对两个答案的偏好差。两者在初始化时完全相同，margin 就为 0，此时损失对该 margin 的导数为 −0.5。仅凭接近零的 margin 或接近 log 2 的 loss，无法判定模型参数梯度消失。[损失函数](../src/agentlab/dpo/loss.py)

就执行结果而言，第 2 轮调整 β 后仍通过 11 题。第 3 轮的偏好损失下降，dev 修复数却减至 10。若继续研究负样本选择，可在固定 reference、预算和 dev 题目的前提下比较配对策略。[FEST 实现笔记](FEST-ANALYSIS.md)
