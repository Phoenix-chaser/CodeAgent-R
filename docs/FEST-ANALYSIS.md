# FEST 论文分析与 DPO 自适应 β 改进

> 来源：[FEST: Few-Shot Demonstration-Guided RLVR](https://arxiv.org/abs/2605.15012)

FEST 在 RLVR 框架下用仅 128 条随机 SFT 示范即可显著提升 RL 性能。
三个关键组件：监督信号（SFT on 示范）、策略内信号（模型 rollout 作为
DPO rejected）、衰减权重（防过拟合）。

### 半在线 DPO 梯度分解

```
∇L_DPO = -β · σ(β(r⁻-r⁺)) · (∇log π(y⁺) - ∇log π(y⁻))
```

> **半在线 DPO ≈ 带负奖励的 REINFORCE + 加权 SFT**

当策略已偏好 y⁺ 时 σ→0，梯度趋零。这就是 DPO 失败的理论根因。

### 自适应 β（论文 Eq. 5）

| 情形 | β | 作用 |
|---|---|---|
| 不可解任务（全部 N 个 rollout 失败） | β₁（大，0.1） | 强力向教师学习 |
| 可解但本次失败 | β₂（小，0.01） | 温和纠正 |
| 本次通过 | β₃（极小） | 仅正则化 |

## 2. DPO 失败的 FEST 解释

DPO 实验 实测：`preference_accuracy=1.0` 全程不变、`reward_margin≈0`、
`loss≈log(2)` — 即策略在初始就已排好序（教师 > 自身失败），DPO 无信号。

根因映射：

| FEST 要素 | DPO 实验 现状 | 差距 |
|---|---|---|
| 自适应 β | ❌ 统一 0.1 | **核心缺失** |
| 群组统计 | ❌ 不知每题通过率 | 需要 N 次采样的群组数据 |
| Hard negative | ❌ 随机选失败 | 应选模型最自信的错误 |
| 衰减权重 | ❌ 无 | 多 epoch 需要加 |
| Token 级一致性 | ❌ 序列级 | 兼容未来 GRPO 时需要 |

## 3. 改进方案（本仓库已实施 A+B）

### 改进 A：自适应 β [P0]

`generate_preferences.py` 新增 `--samples N` 群组采样，记录每题全部 N 个
采样的通过/失败到 `group_solvable` / `group_pass_rate` 字段。
`hf_dpo.py` 训练循环读取该字段，对不可解任务用 `--beta-unsolvable`
（默认 0.1），可解任务用 `--beta-solvable`（默认 0.01）。

### 改进 B：Hard-Negative 选择 [P1]

对每题的多个失败输出，用 AST 指纹聚类找出**最常出现的失败模式**，
选该模式的一个实例作为 rejected。模型最自信的错误才是它最需要
"忘掉"的。

### 改进 C：衰减权重 [P2，暂缓]

多 epoch 场景下 DPO 权重从 1.0 衰减到 0.01。单 epoch 不涉及，
待 A+B 验证有效后实施。

### 改进 D：Token 级 DPO 变体 [P3，暂缓]

FEST-GRPO 将 DPO 分解为 token 级加权 SFT + 负优势 GRPO，消除
序列级与 token 级的梯度量级差（10² vs 0.1）。需要与 GRPO 联合训练
时实施。

### 改进 E：联合训练 [P4，暂缓]

FEST 同时在两个数据集上训练（GRPO on 答案数据 + DPO on 示范数据），
而非当前的顺序 SFT → DPO。需架构重构，在 A+B+D 之后考虑。
