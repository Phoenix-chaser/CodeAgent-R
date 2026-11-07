# CodeAgent-R 复审修复记录

复审日期：2026-11-01  
项目目录：`<本机项目目录>\CodeAgent-R`

## 1. 结论与实际验证

本轮已直接修改代码、补充回归测试、对齐文档。当前可以验收为经过离线验证的代码 Agent 评测与后训练实验基础设施，尚不能验收为完成真实语言模型训练和在线 RL 的系统。

最终全量验证结果：

```text
216 passed in 12.77s
```

执行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest tests -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m agentlab --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m agentlab doctor
git diff --check
```

另外通过 `yaml.safe_load` 解析了状态文件和两个实验/训练合同。测试未出现 warning。验证环境为 Linux、Python 3.14.4、torch 2.14.1+cpu；本轮没有安装依赖、调用真实 API、租用 GPU 或执行真实模型训练。没有验证全新虚拟环境安装，也没有验证原生 Windows 执行。

当前执行测试使用 Bash、POSIX 进程组和 Linux 工具，Windows 用户建议在 WSL2/Linux 环境完成执行与训练验收。项目路径位于 Windows 磁盘不代表运行后端已经跨平台。

## 2. 本轮修复的实际缺陷

| 模块 | 修复前的问题 | 修复后的行为与证据 |
|---|---|---|
| DPO loss | 极端负 margin 经 sigmoid 再取 log 得到 inf，梯度可变成 NaN | 改为稳定 logsigmoid；极值 loss/梯度有限的回归通过 |
| DPO 全 mask | 返回新建标量，可能脱离原计算图或设备 | 返回与输入图相连的零值；设备、dtype 和零梯度行为通过验证 |
| DPO reference | reference tensor 可能带梯度 | reference 明确 detach；测试确认梯度不流入 reference |
| DPO 参数更新 | 仅证明 loss 可调用不足以说明训练生效 | toy 更新测试确认参数实际变化、梯度有限；仍不代表真实语言模型 DPO |
| DPO collator | prefix 与 candidate 分开渲染导致系统前缀重复 | 完整消息一次渲染；目标 mask 与共享上下文保持一致 |
| RL 来源校验 | runner 改写版本/checkpoint；直接构造绕过 add 校验 | 构造、add、消费时校验 task/source/split/checkpoint/policy；拒绝不匹配来源 |
| RL 多轮动作 | 只使用第一轮 logp/mask，长度不一致可被 zip 截断 | 全轮次汇总前逐轮校验形状；第二轮动作也影响 loss |
| RL 更新声明 | 浮点数学检查易被理解成模型更新 | 日志明确 math_check、update_applied=false，并区分 identity/provided logps |
| RL KL | 估计量解释与采样条件不清晰 | 使用非负采样项，说明仅在 new-policy 采样条件下期望等于对应 KL |
| RL 成本 | 已知费用与未知费用相加时把未知当零 | 总额保留 None，另记已知小计与未知条数 |
| 分支模型预算 | candidate_fn 在准入前直接调用，绕过模型次数上限 | 候选先准入；预算耗尽时 callback 不执行；候选和续跑共享账本 |
| 运行时结算 | 请求/工具只预留不结算；finish 请求未被计入 steps | 每次实际调用结算次数，finish 也算模型请求，回调异常保留发生次数 |
| 操作标识 | 重复工具 ID、再次调用 run 可能复用预算操作 | 每次执行具有独立账本操作 ID；重复外部 ID 不能免扣预算 |
| 账本释放 | 预留 10、实际 3 时仅释放 3，留下错误冻结额度 | 释放该操作原始预留，再记录实际发生额；不误释放其他操作额度 |
| 幂等结算 | 未预留 seed 重放可重复计费 | 已结算操作重放不重复入账；未知结算保留未知状态及预留 |
| 预算中断轨迹 | 后续工具被拒绝时，已完成工具的观测丢失 | step 返回部分观测及 budget_exceeded，run 保留已完成工具历史 |
| 分支统计 | initial_budget 实为已耗额度；费用按声明调用数量计算 | initial_budget 为剩余预算；候选与续跑按账本差值记模型/工具次数 |
| 偏好数据 | invalid、infra、partial 可能被当作失败答案 | rejected 仅接受 strict_fail；预算拒绝分支也不进入当前偏好构造 |
| API 估价 | 缺 output_tokens 却按零估价；模型不同仍套供应商旧费率 | 缺任一用量、未配价或模型不匹配均保留 unknown；显式模型级来源价表 |
| API 请求 | SDK 隐式重试不受外部次数预算管理 | 设置 timeout、max_retries=0；重试需由上层作为新请求调度 |
| API 错误 | 文本模糊匹配分类且可能回显供应商敏感信息 | 依据 HTTP 状态与异常类型分类，不回显原始异常正文 |
| CLI/依赖 | 文档中的 python -m agentlab 入口不存在；离线测试缺 torch 声明 | 增加模块入口；dev 包含 torch，api extra 提供可选 SDK |
| 文档状态 | 列出未实现训练命令，将资源准备与训练完成混淆 | README/英文版/资源指南和状态文件统一区分已实现、契约验证、待实现 |

核心回归测试位于：

- `tests/test_runtime.py`：预算结算、重放、并发准入、隔离。
- `tests/test_p04_runtimes.py`：模型次数、finish、重复工具 ID、已执行观测保留。
- `tests/test_p06_branching.py`：候选/续跑共同预算、实际费用计数、偏好过滤。
- `tests/test_p08_dpo.py`：稳定 loss、图连通、reference 隔离、参数变化、模板一致性。
- `tests/test_p09_rl.py`：来源校验、所有轮次动作、形状错误、KL 与 unknown 费用。
- `tests/test_providers.py`：43 项假客户端测试，无网络请求。

## 3. 必须保留的实现边界

### 执行和计费

1. 请求数、工具次数预算已在本轮验证。跨模型回调与工具的总 wall_seconds 强制截止尚未打通，不能宣称具有完整运行时熔断。
2. 当前没有美元上限接线。provider 超时设置也不等于整个 episode 的截止时间。
3. 分支费用中的 model_requests/tool_calls 是调用次数；USD 保留 None。它们不能自动推导 token 或真实账单。
4. ModelPricing 是输入/输出统一单价估计，缓存折扣、阶梯价、税费等需要额外字段与账单校验。
5. provider 尚未串联完整 Agent episode：需要完成消息转换、tool-call JSON 校验、工具响应回填、usage 入账、原始响应关联和持久化。
6. 当前进程工作区隔离不等于通过安全验收的容器沙箱；真实不可信代码执行仍需容器、资源和网络边界验收。

### 训练与研究证据

1. SFT/DPO 已验证的是 stub tokenizer、toy 模型和局部数学路径。真实 tokenizer、因果语言模型、LoRA/QLoRA trainer、训练 checkpoint 和重载评估仍需实现。
2. RLRunner 没有模型 forward/backward、optimizer step 或权重同步。函数名 train_step 保留兼容，不代表已训练。
3. 缺省 new/ref logps 的路径明确标为 identity_logps，仅作数学检查。真实训练不能以这种回退代替模型重新计算概率。
4. RL 分组尚缺完整环境、可见前缀和剩余预算指纹。现有 task/checkpoint 校验不能替代完整同状态证明。
5. 项目只有 6 个 smoke 任务。暂时没有独立未见仓库上的训练收益、统计置信区间或正式横评结果。
6. pi 未安装，真实 pi JSON 协议和能力尚未验收。

## 4. 下一阶段的实施顺序与验收条件

### A. 接通一个真实 API episode

先完成 provider→受控运行器的适配，明确“教师/商业基线”身份，再运行少量可审计样例。

验收：可复现一次读取→修改→测试→结束；预算拒绝发生在请求前；保存模型版本、请求/响应关联、真实 usage、未知费用标记和最终私有 grader 结果；网络故障与任务失败分开。

### B. 补齐状态指纹与数据产物

统一记录任务版本、fixture/snapshot hash、环境镜像、公开 prefix hash、工具 schema、剩余预算、policy/reference/tokenizer revision。将 source_group 划分写入 dataset manifest，拒绝 train/test 混入和未来信息泄漏。

验收：同状态偏好对能由 manifest 重放；任一关键状态不一致都拒绝；数据去重与划分规则有明确证据。

### C. 实现真实 SFT 纵向闭环

接入真实 tokenizer 和可训练模型，验证 token 级监督 mask；完成 optimizer、梯度累积、checkpoint、重载与固定测试集 Agent 评估。

验收：可展示实际参数变化、有限梯度、监督 token 统计、保存前后输出一致性，以及基座/SFT 在相同预算上的结果。先跑小型模型和短序列，再决定是否上 7B QLoRA。

### D. 增加 DPO 对照

真实 reference policy 固定；同状态 chosen/rejected 使用相同模板和上下文，缓存 logp 必须绑定模型、tokenizer 与样本 hash。

验收：SFT 与 SFT+DPO 使用相同评测任务、模型推理设置和预算；至少报告成功率、无效动作率、恢复成功率、费用和失败类型。比较训练成本时同时报告样本数、监督 token 和更新步数。

### E. 最后接在线 RL

rollout 必须由正在训练的 policy 或其明确 old-policy 快照产生；训练端计算可微 new logp，固定 reference，记录采样策略与更新前后版本。实现梯度裁剪、异常批次处理、零方差组统计、checkpoint 恢复。

验收：一次完整 rollout→reward→advantage→backward→optimizer→权重同步后，新轮采样使用新权重；保留两轮以上的证据。商业教师轨迹可作离线训练数据，不能标成自己模型的 on-policy rollout。

