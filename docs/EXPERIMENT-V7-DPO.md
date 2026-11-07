# v7 DPO 实验：实现完整、效果未达（2026-11-07）

P08 从"离线契约"升级为真实实现并完成端到端实验：偏好构造 → DPO 训练 →
协议内选型。结果为负，根因清晰且可复用。

## 实现（commit 6b7c292，468 项测试）

- `scripts/generate_preferences.py`：学生模型（SFT adapter）在 train 任务
  温度采样 → 三层判分 → strict_fail 与教师 strict_pass 配对；共同状态 =
  同一公开 prompt。本轮实测：64 题 × 6 采样（51 分钟），学生通过率
  207/384=54%，产出 50 对（每题上限 1 对）。
- `src/agentlab/training/hf_dpo.py`：policy = Base(NF4) + SFT adapter 继续
  训练；reference = 同 adapter 冻结；chosen/rejected 共享字节级一致监督
  前缀；偏好准确率与 reward margin 全程记录；每 epoch 不可变快照。
- β=0.1，lr=5e-7，1 epoch（50 步，49 秒）。

## 结果与根因

| 模型 | dev pass@1（20 题） |
|---|---|
| Base | 50% |
| DPO | 55% |
| SFT（选中保持） | **60%** |

训练日志：`preference_accuracy=1.0` 全程不变，`reward_margin≈0`，
`loss≈log 2` 无下降。**偏好关系在初始化即被满足**：policy 与 reference 同源，
初始 margin = β·(logp_教师 − logp_学生失败) > 0——SFT 本来就更偏好教师解而
非自己的失败采样，sigmoid(δ)≈0.5，梯度趋零，DPO 近似 no-op。dev −1 题在
噪声内。

按 dev 选型协议（未超 SFT 不上线），DPO 不进入 test 终评，SFT 保持选中。

## 方法论结论

1. `chosen=教师 vs rejected=模型自身失败` 的配对对 DPO 是"已答对的题"，
   无梯度信号。有效偏好对需要 **hard negatives**：策略相对更偏爱的失败，
   例如同一策略分布内的 pass/fail 采样对，或按 logp 排序挑选的失败。
2. 防退化选型再次生效：DPO 未超过 SFT 时不上线，与多 seed 协议中
   seed 43 保留 Base 同一机制、同一逻辑。
