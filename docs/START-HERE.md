# CodeAgent-R 当前操作入口


## 现在先做什么

1. 使用新版代码评测现有 第2轮 adapter 的 dev 表现，判断它是否值得保留。
2. 使用已重验的 `data/experiment-第3轮-revalidated/`，运行一次短训练；默认保存每个 epoch。
3. 运行 `select_checkpoint.py`，在 dev 执行测试上比较 Base 和所有 epoch；平局或退化保留 Base。
4. 基于 dev 结果决定是否扩大独立任务覆盖；正式收益使用新的、未看过的 test 确认。

本次无需重新付费采集，一张完整 24GB 的 GPU 足够复跑。数据修复、训练、选择的完整可复制命令见 [GPU 实验手册](POSTTRAIN-GPU.md)；

## 新项目与后续研究

从零构建新任务集时，参考 [GPU 实验手册](POSTTRAIN-GPU.md) 的来源组划分、公开输入、隐藏测试审计和实验矩阵。内置 84 题仍是 pilot，重复教师答案不能算独立训练任务；原 14 题仅作 smoke，已分析过的 12 道 test 只作诊断。

DPO/GRPO 仍是离线组件，真实在线 RL 和完整工具交互 Agent 尚未验收。当前可以展示可复现的数据/训练/执行评测工程；改版后的模型收益等待 GPU 实测。
