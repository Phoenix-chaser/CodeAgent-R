# FEST 实现笔记

[三轮 DPO 实验](EXPERIMENT-DPO.md) · [实验成绩单](EXPERIMENT-RESULTS.md)

FEST 研究如何用少量示范引导带可验证奖励的强化学习。CodeAgent-R 在离线 DPO 中借用了按题目采样结果调整监督强度的思路，同时用失败样本的语法结构选择负例。[FEST 论文](https://arxiv.org/abs/2605.15012) · [DPO 论文](https://arxiv.org/abs/2305.18290)

## 一组采样如何变成训练信号

| 同一道题的采样结果 | 数据字段 | 本仓库的处理 |
| --- | --- | --- |
| 全部失败 | `group_solvable=false` | 从失败样本中选负例，默认 β=0.1 |
| 有通过也有失败 | `group_solvable=true` | 从失败样本中选负例，默认 β=0.01 |
| 全部通过 | 没有失败候选 | 跳过该题的偏好对 |

生成器以 `--samples` 设置每题采样数，在偏好记录里保存 `group_pass_rate`。有多个失败输出时，生成器按抽象语法树指纹计数，取出现最多的结构中的一个实例。频次描述的是这批采样里重复出现的错误，模型对该答案的概率需要另行计算。[偏好生成脚本](../scripts/generate_preferences.py)

## 在训练端核对什么

`--beta-unsolvable` 和 `--beta-solvable` 分别设置上述两类偏好对的系数。policy 从 SFT adapter 开始，`--reference-adapter` 指定冻结的参照 adapter，省略时使用 Base。该设置决定了计算偏好差时减去哪一个模型的对数概率。[训练参数](../scripts/train_dpo.py) · [损失计算](../src/agentlab/dpo/loss.py)

在 20 道 dev 题上，统一 β 和按组设置 β 的两轮都通过 11 题。切换到 Base reference 后通过 10 题。当前记录支持比较各轮执行得分，单独归因给某个组件还需要控制其余设置的消融实验。[实验记录](EXPERIMENT-DPO.md)

原方案还讨论了跨 epoch 衰减权重、token 级偏好目标以及与 GRPO 联合训练。这三项保留为研究方向。
