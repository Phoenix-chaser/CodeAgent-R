# CodeAgent-R 审查与优化计划

> 更新（2026-11-01）：本文件保留首轮审查背景，其中部分缺陷已修复，不能再当作当前待办。
> 本次直接修改后的代码状态、216 项测试结果、剩余工作和 API/GPU 建议见 [复审修复记录](docs/REVIEW-FIXES.md)。
> requirements/YAML/模块入口/DPO 梯度与数值稳定性等问题已处理；真实语言模型训练和在线 RL 仍未完成。

## 0. 首轮审查结论（历史）

当前仓库已经完成了 Agent 评测、轨迹契约、失败恢复数据结构、assistant-only 监督 mask、共同状态偏好构造和 GRPO 风格数学对象的离线骨架，并且离线测试可以通过。

当前实际测试结果：

```text
PYTHONPATH=src python3 -m pytest tests -q
109 passed
```

这些测试主要验证数据契约、边界条件、隔离逻辑和 toy 模型数学检查，不能证明真实 7B 模型已经完成 SFT、DPO 或在线 RL。

当前项目不能对外宣称：

- 已完成 7B Code Agent 的 SFT 训练；
- 已完成可更新模型参数的 DPO；
- 已完成真正的在线 GRPO/在线 RL；
- 已接入商业模型 API 并记录真实 token、费用和事件流；
- 已完成 pi JSON 协议和 usage 兼容验证；
- 已证明某种训练方法提高了未见仓库的修复成功率。

当前最准确的项目定位是：

> 面向代码 Agent 的执行反馈评测、失败恢复数据构造和后训练实验基础设施，已完成离线训练契约与数学验证，正在补齐真实模型训练和运行时闭环。

## 1. 当前问题清单

### P0：文档和仓库状态不一致

涉及文件：

- `README.md`
- `README_EN.md`
- `docs/RESOURCE-GUIDE.md`
- `docs/project-status.yaml`
- `pyproject.toml`
- `src/agentlab/cli.py`

问题：

1. README 仍描述旧测试数量和旧阶段。
2. 文档列出的 `run-pilot`、`train-sft`、`report` 等命令，当前 CLI 没有注册。
3. README 要求安装 `requirements.txt`，仓库中不存在该文件。
4. `docs/project-status.yaml` 当前无法被 `yaml.safe_load` 正确解析。
5. 文档容易让读者误以为真实 SFT/DPO/RL 已完成。

修复要求：

- 文档明确区分 `implemented`、`contract-only`、`planned`。
- README 的安装命令必须能在干净环境执行。
- 每个文档命令都必须对应真实 CLI 子命令；未实现命令移到 roadmap。
- 状态文件加入 CI YAML 解析检查。
- 测试数量和阶段编号由脚本生成，避免手工过期。

验收：

```text
python -m agentlab --help
python -m agentlab doctor
python -c "import yaml; yaml.safe_load(open('docs/project-status.yaml', encoding='utf-8'))"
PYTHONPATH=src python -m pytest tests -q
```

### P1：实现真实最小 SFT

当前 `src/agentlab/training/entry.py` 只是 `Embedding + Linear` 的 CPU toy forward/backward。它只能验证最小数学链路，不能证明真实训练。

需要补齐：

1. `transformers` tokenizer 和 causal language model。
2. `datasets` 或等价的数据读取层。
3. `peft` LoRA/QLoRA 配置。
4. `accelerate` 或明确的单卡训练入口。
5. optimizer、scheduler、gradient accumulation、mixed precision。
6. checkpoint 保存、恢复和训练状态记录。
7. 训练集、dev 集、test 集严格分离。
8. 训练后重新运行 Agent 任务，而不是只报告 loss。

建议新增模块：

```text
src/agentlab/training/hf_model.py
src/agentlab/training/tokenization.py
src/agentlab/training/sft_trainer.py
src/agentlab/training/checkpoint.py
src/agentlab/training/evaluate_checkpoint.py
configs/train/sft-smoke.yaml
scripts/train_sft.py
```

#### SFT mask 的实现要求

- system、user、tool observation 默认 mask 为 0。
- assistant 的自然语言回复和合法 tool call mask 为 1。
- 失败历史可以出现在可见上下文中，但不能自动成为正向目标。
- tokenizer 必须能得到消息边界，不能依赖字符串长度猜 token 边界。
- 截断不能破坏 JSON tool call 或消息配对。
- sequence packing 必须有 attention boundary，禁止不同任务互相注意。
- 至少测试一条真实 tokenizer 的逐 token mask 可视化样例。

#### P1 验收标准

- 使用真实开源基座模型完成一次可复现的单卡 LoRA smoke test。
- 训练前后 adapter 参数确实发生变化。
- checkpoint 可以保存、重新加载并得到相同 tokenizer/template 配置。
- dev 上至少能产生合法工具调用。
- 训练日志记录实际 token 数、有效监督 token 数、峰值显存、耗时和 seed。
- 不能把 toy 模型测试当作 7B 训练结果。

### P2：修复并实现真实 DPO

当前 `src/agentlab/dpo/loss.py` 在 `sequence_logp` 中使用 `.item()`，会切断梯度；当前 micro step 只返回 loss 指标，没有真实 optimizer update。

需要实现：

1. chosen/rejected 的 tensor log-prob 保持计算图。
2. reference model 冻结，policy model 可训练。
3. chosen/rejected 使用相同 task、workspace、prefix、tools 和剩余预算。
4. 只对 assistant action token 累计 log-prob。
5. tool observation 不进入 policy 输出概率。
6. 候选动作的 token 和执行成本计入共同预算。
7. 支持 batch、padding mask、梯度累积和 checkpoint。
8. 训练后重新运行状态恢复评测。

偏好数据必须保存：

```text
task_id
source_group
split
workspace_snapshot_hash
visible_prefix_hash
tool_schema_hash
remaining_budget
chosen_action
rejected_action
chosen_terminal_result
rejected_terminal_result
label_confidence
policy_version
```

不能把来自不同失败状态的“成功轨迹”和“失败轨迹”直接拼成 DPO 对。

#### DPO 验收标准

- 一个可解释 batch 能产生非零梯度。
- optimizer step 后 policy 参数发生变化，reference 参数不变。
- chosen/rejected 的共同状态契约检查失败时，数据被拒绝。
- 训练前后在固定恢复状态上重新执行，而不是只看 DPO preference accuracy。
- 报告长度差异、候选成本、低置信偏好和偏好来源。

### P3：实现真实在线 RL/GRPO

当前 `src/agentlab/rl/runner.py` 没有可训练 policy、optimizer、权重同步和真实 old-policy 概率；`new_logps = old_logps = ref_logps` 使 ratio 恒为 1、KL 恒为 0。

正确闭环应为：

```text
当前 policy 生成同一恢复状态的多条 rollout
→ 工具执行和私有 grader
→ 计算终局奖励和组内 advantage
→ 使用 old-policy log-prob 做裁剪更新
→ 更新 policy 权重
→ 记录 policy_version
→ 用新 policy 继续采样
```

必须补齐：

- 可训练 policy model；
- optimizer、KL 和 entropy 配置；
- old/new/reference policy 的一致 tokenizer 与 template；
- 只对本次 rollout 新生成的 assistant token 计算 policy loss；
- rollout 与 policy version 绑定；
- 权重更新后的刷新或明确的 stale rollout 策略；
- zero-advantage group 统计；
- done、timeout、budget exceeded、environment failure 的区分；
- checkpoint 恢复 optimizer、policy version 和采样配置。

第一版不要直接从 50 步正式 RL 开始，应按以下顺序：

1. 一个 toy environment 的单步更新。
2. 一个真实代码任务的单步候选动作更新。
3. 多轮恢复状态的 4 条 rollout 分组。
4. 少量训练任务的 10 步 pilot。
5. 只有奖励、梯度和任务成功率都正常后，才扩展到 50 步。

## 2. Runtime 和数据链路

### P4：实现真实 API provider adapter

当前 `ControlledModelRuntime` 只注入 `candidate_fn`，没有真实商业 API 接入。应新增统一 provider 接口：

```text
src/agentlab/providers/base.py
src/agentlab/providers/openai_compatible.py
src/agentlab/providers/<provider>.py
src/agentlab/runtime/api_runtime.py
```

统一记录：

- provider 和 model name；
- model revision 或服务版本（如果可用）；
- request id；
- input/output/cache token；
- latency；
- retry 和 error type；
- estimated cost；
- tool call JSON；
- stop reason；
- 原始事件与规范化事件的 hash。

API key 只从环境变量或本地 secret store 读取，不能写进日志、轨迹或 Git。

API 的正确用途：

- 生成教师轨迹；
- 生成 SFT 候选数据；
- 生成 DPO 候选并由私有 grader 判分；
- 作为强模型 baseline；
- 验证 Agent 任务、工具协议和评测平台。

API 不能直接完成：

- 对闭源模型做 LoRA/QLoRA；
- 修改闭源模型参数；
- 把闭源模型输出直接称为自己的 on-policy GRPO rollout；
- 证明自己的 checkpoint 能力提升。

### P5：修正轨迹分支和预算账本

`src/agentlab/trajectories/branching.py` 当前存在以下问题：

- `remaining_budget` 被错误记录为 `consumed_budget`；
- 分支预算硬编码，未从父 episode 继承；
- token 和美元费用没有接入真实 usage；
- `RolloutGroup` 对 source group、split、环境、工具 schema 和剩余预算约束不足。

修复要求：

1. 父 episode 保存完整 `budget_state`。
2. 分支从父快照复制 workspace、工具 schema 和剩余预算。
3. 每次模型调用和工具调用原子地更新账本。
4. unknown cost 不能当作 0。
5. 同状态分组必须比较 snapshot hash、prefix hash、tool schema、environment、remaining budget 和 policy version。
6. 账本测试覆盖超时、重试、取消、重复事件和分支合并。

## 3. API 与云 GPU 的资源方案

### 3.1 最实用的混合方案

```text
商业 API：教师轨迹、强模型 baseline、候选生成
云 GPU：自己的开源模型 SFT、DPO、在线 RL
本地 CPU：任务、sandbox、grader、数据清洗、统计和单元测试
```

这是当前项目最合理的资源分工。

### 3.2 哪个更好用

如果目标是先把项目跑起来，API 更方便：不用处理 CUDA、模型下载、显存、推理服务和驱动。但是 API 只能输出结果，不能让你训练闭源模型。

如果目标是证明自己真正实现了 SFT、DPO 和在线 RL，云 GPU 是必需的。云 GPU 不是 API 的替代品，而是训练可加载开源基座模型的执行环境。

### 3.3 推荐顺序

1. 本地 CPU 完成任务、判分、轨迹和数据契约。
2. 接入一个真实商业 API，先采集 20–50 条轨迹并验证 usage/cost。
3. 租用单张 24GB GPU，先完成短上下文、小模型或 7B QLoRA smoke test。
4. SFT 真实 checkpoint 可用后，再做 100 对左右的 DPO pilot。
5. 在线 RL 最后做；rollout 必须来自自己的 policy，而不是直接来自商业 API。

### 3.4 对原估算的修正

原先的 30–60 分钟 SFT/DPO 估计可以作为小规模 pilot 目标，但必须记录平均输入长度、输出长度、有效监督 token 数、batch、梯度累积和实际吞吐。

P09 的“50 步 × 4 rollouts”是 200 条 rollout，不是 200 次简单 API 请求。若使用商业 API，它们只能作为教师或离线数据，不能直接证明自己的模型做了 on-policy GRPO。

P10 如果比较基座、SFT、DPO、RL 四个 checkpoint：

```text
20 test × 3 repetitions × 4 versions = 240 episodes
```

60 episodes 只对应单一模型版本的评测。

## 4. 实验设计要求

主问题应改成可证伪形式：

> 在相同模型、工具、任务、有效监督 token 预算和推理预算下，失败恢复轨迹训练是否提高未见仓库的严格修复成功率？

至少保留以下模型对照：

| 编号 | 模型 | 作用 |
|---|---|---|
| M0 | 原始基座 | 能力下界 |
| M1 | 最终补丁 SFT | 答案监督基线 |
| M2 | 完整成功轨迹 SFT | 普通过程监督基线 |
| M3 | 失败恢复 SFT | 核心方法 |
| M4 | M3 + 同状态 DPO | 偏好优化 |
| M5 | M4 + 在线 RL | 高阶扩展 |

所有模型必须使用相同：

- 工具协议；
- workspace 初始状态；
- 测试和隐藏 grader；
- 最大 token、步数、时间和费用预算；
- 评测任务 split；
- 重复次数和随机种子记录规则。

必须报告：

- 严格 success@budget；
- 固定失败状态 recovery success；
- tool-call 合法率；
- 无效循环率；
- 回归错误率；
- 实际输入/输出 token；
- 成功成本；
- p50/p95 延迟；
- 基础设施失败率；
- 按任务和 source group 的配对差异与区间。

训练 loss 下降、DPO preference accuracy 上升或 RL reward 上升，都不能单独作为任务能力提升证据。

## 5. 最终完成定义

项目达到可写进简历的“真实训练版”，至少需要：

- 一个真实开源基座模型；
- 一次真实 SFT checkpoint；
- 一次真实 DPO checkpoint；
- 一次可解释的在线 RL pilot，或明确记录失败原因；
- 真实 API usage/cost 事件；
- 真实 checkpoint 的固定 dev/test 评测；
- 训练和评测配置可恢复；
- 所有文档命令可执行；
- 没有把外部 API 输出冒充自己的训练结果；
- 没有把 toy backward 冒充真实模型训练。

如果最终没有云 GPU，项目应改名并改定位为：

> 执行反馈驱动的 Code Agent 评测与失败恢复策略实验平台。

此时可以保留 API 轨迹采集、grader、成本统计、候选选择和恢复策略实验，但不要声称完成 SFT、DPO 或在线 RL 参数训练。

## 6. 当前简历表述建议

在真实训练完成前，建议写：

> 设计代码 Agent 执行反馈评测平台，构建隔离沙箱、可信测试判分、失败恢复轨迹和同状态偏好数据管线；实现 assistant-only SFT mask、DPO/GRPO 训练契约与离线数学验证，完成 109 项自动化测试。

完成真实训练并有可靠结果后，再补充：

> 使用开源代码模型完成 LoRA SFT、同状态 DPO 与执行奖励在线 RL，比较未见仓库上的严格修复成功率、恢复率、成本和稳定性。

不要预先填写未测出的提升百分比、成本下降或成功率提升。

