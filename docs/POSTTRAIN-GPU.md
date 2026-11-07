# GPU 后训练实验手册

基础流程编写于 2026-11-03，其后更新了 epoch 保存与 dev 执行选择。最终结果见 [v6 实验结果](EXPERIMENT-V6-RESULTS.md)。本文保留数据、训练、选择与评测的通用流程。

## 1. 先解释 7/14 → 6/14

这次变化是 **−1 题、−7.14 个百分点**；相对成功率变化为 −14.29%，两种口径不能混写。旧 14 题同时参与候选采集和评测，并被反复查看，属于污染集合。旧 GPU 评测启用随机采样却没有固定 seed，重复评测通过 `any()` 汇总，与单次成功率含义不同。仅凭这张表无法区分训练损伤、随机波动、数据问题或评测协议问题。

审阅代码发现的可修复问题包括：跨文件任务没有向模型提供依赖文件；采集器把所有任务写成 train；训练未用独立 dev 选择 checkpoint；小数据尾批可能没有执行优化器更新；旧日志的 loss 和累积计数耦合。当前入口已针对这些路径改进，旧表需要重新建立可比基线。

旧代码提取器还会删掉状态机主文件开头的常量表，导致正确代码在评分前被破坏；现已保留完整合法 Python，包括中文字符串及字符串中的 Markdown 围栏。部分任务的公开需求与隐藏测试要求不一致，也已把原本隐藏的接口约束明确到公开指令中。隐藏测试内容仍不进入 prompt。这些修复改变了评测协议，不能把新分数相对旧分数的增长归功于 SFT。

保留旧 `sft-v2`、训练日志和原始数据，不覆盖它们。新协议的分数不能直接减去旧协议分数；Base 和 SFT 都要在同一新协议下重跑。

## 2. 明确硬件和依赖

用户当前确认的是 **24GB 单卡 GPU**。早期文档中的显存份额记录属于旧环境，不代表现在的实际分配。判断显存看 `nvidia-smi` 与 PyTorch 可见容量，不能只看卡名。

| 配置 | 推荐起点 | 需要实测 |
|---|---|---|
| 完整 24GB 单卡 | Qwen2.5-Coder-7B-Instruct、NF4、长度 2048、batch 1、累积 8 | 完整累积窗口、dev forward、保存重载的峰值显存与吞吐 |
| 约 12GB 显存份额 | 先用 1.5B 或 3B、NF4、长度 1024、batch 1 | 不承诺 7B 在同配置可行；检查长样本丢弃率 |

累积步数控制有效 batch，不降低单个 microbatch 的显存。NF4 加载也不等于下载文件只有 4-bit 大小。显存不足时先检查其他进程，再降低长度或模型规模；不要静默跳过 OOM 样本后继续声称数据配方相同。

建议 Linux/WSL2、Python 3.10–3.12，在独立环境安装与驱动相容的 CUDA PyTorch，再执行：

```bash
python -m pip install -e ".[dev,train,api]"
python -m pip check
python -m pytest tests -q
python -m agentlab doctor
nvidia-smi
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0).total_memory / 2**30)
assert torch.cuda.is_bf16_supported(), "bf16 unavailable"
values = torch.randn(256, 256, device="cuda", requires_grad=True)
values.square().mean().backward()
assert torch.isfinite(values.grad).all()
torch.cuda.synchronize()
print("CUDA forward/backward passed")
PY
```

`train` extra 的范围为 torch 2.4–2.x、transformers 4.46–4.x、PEFT 0.13–0.x、accelerate 1.x、bitsandbytes 0.44–0.x。这是依赖约束，不是已验收锁文件；真实 GPU 环境安装通过后保存 `python -m pip freeze`。本次修改环境没有 `nvidia-smi`，没有运行 GPU 训练。

## 3. 先扩建并冻结任务

正式实验目标为 **500+ 独立 train 任务、独立 dev、100+ held-out test 任务**。这是后续建设目标；14 个历史 smoke 已污染，新编写的 24 题、12 来源组 pilot 仅用于验证闭环。不能通过更换 task_id、变量名、常量或重复生成答案来凑数量。

第一次运行可以导出 pilot 作为下面的任务池：

```bash
python scripts/export_pilot_tasks.py --output data/task-pool.jsonl
```

每个来源组中的两题一起进入同一 split，来源组数量远小于题数。即使 pilot 的 test 分数上涨，也只能报告为小样本流程试验，不能写成已证明真实仓库泛化提升。

导出脚本先执行参考修复与原缺陷的判分审计，成功后写入 JSONL 和同名 `.audit.json`，并拒绝覆盖已有文件。本次 24/24 参考修复通过、24/24 原缺陷失败证明任务可用于流程验证，不是模型能力分数。

任务池应覆盖边界输入、异常传播、资源释放、缓存一致性、接口契约、状态机、配置优先级等类别，每个任务必须有清晰公开需求、缺陷代码、必要依赖、公开诊断测试、私有边界/回归测试及参考修复。参考修复通过测试、缺陷版本暴露预定问题之后才接纳。保留任务来源、许可证、仓库/commit、缺陷类型和人工审阅记录。

同一仓库、模板、近重复函数、变体及派生恢复状态放在同一 `source_group`。来源组应真实反映数据谱系，不能为了跨 split 随机改写。公开数据也可能进入基座预训练，公开 benchmark 分数不等于完全无污染。

外部任务 JSONL 使用项目任务规格，核心字段为 `task_id`、`source_group`、`split`、`bug_family`、`difficulty`、`instruction`、`module`、`buggy`、`fixed`、`public_test`、`private_tests`，可用 `extra_files` 提供依赖代码。目前仅支持平铺的 Python 文件名，私有测试文件名以 `test_` 开头。当前流程输出主文件的完整修复代码；依赖文件作为只读上下文。需要修改多个文件的任务应先扩展输出协议与 grader，再纳入任务池。

先冻结任务，再采集答案：

```bash
python scripts/prepare_data.py \
  --tasks-file data/task-pool.jsonl --split-tasks \
  --output-dir data/experiment --dev-ratio 0.15 --test-ratio 0.15 --seed 42
```

产物为 `data/experiment/tasks.jsonl`、`split-manifest.json` 与 `audit-manifest.json`。冻结前会运行参考修复及缺陷复现审计；默认每个测试集超时为 10 秒，可用 `--audit-timeout` 按可信任务需要调整。划分以来源组为单位，实际任务比例可能不同于 70/15/15；查看各 split 的任务数、来源组和类别分布。输出目录必须为空，以保护已冻结的实验。旧 14 题仅保留 smoke，禁止混入新的 train/dev/test。

## 4. 教师候选与数据审计

采集 prompt 与评测共享公开协议：指令、主文件、依赖文件、公开测试。参考修复、隐藏测试、grader 答案不能进入模型输入。教师输出必须经独立 grader 验证；教师名气不能替代测试。

用与账户匹配的标准 API 配置，密钥只放环境变量：

```bash
# 按 collect_data.py 的 --base-url / --api-key-env 参数配置教师 API；
# 密钥只放环境变量，不要写入文件或命令历史。
python scripts/collect_data.py \
  --tasks-file data/experiment/tasks.jsonl --split train \
  --candidates 4 --output data/train-candidates.jsonl
python scripts/collect_data.py \
  --tasks-file data/experiment/tasks.jsonl --split dev \
  --candidates 2 --output data/dev-candidates.jsonl
python scripts/prepare_data.py \
  --input data/train-candidates.jsonl data/dev-candidates.jsonl \
  --tasks-file data/experiment/tasks.jsonl --output-dir data/experiment/sft
```

这些采集命令会发送付费请求；模型可用性和计价以实际账户为准。不要对 test 采集训练目标。`prepare_data.py` 在已有划分下验证、去重并重新执行所有输出样本，输出 `train.jsonl`、`dev.jsonl` 与 `dataset-manifest.json`；不会仅信任候选自带的 `strict_pass` 字段。默认每任务最多保留 1 条，按最小 token 编辑量排序后重放验证，可用 `--max-per-task` 调整。数据准备输出目录必须为空。

检查独立 task/source_group 数、重复/拒绝样本、类别覆盖、教师通过率、截断率、有效监督 token 和来源版本。五百条来自十道题的候选仍然只有十道任务。失败回答可以留作错误分析，但单纯把成功和失败输出拼成一对不等于合法 DPO 数据；需要同状态与可靠标签。

## 5. 训练与 checkpoint 选择

以下是 24GB 单卡 的起点，不是收益保证。设置实际模型 commit 以锁定模型与 tokenizer；每次训练使用新目录，脚本拒绝覆盖非空输出目录。

```bash
export MODEL_REVISION='填入已核实的模型commit'
python scripts/train_sft.py \
  --data data/experiment/sft/train.jsonl \
  --eval-data data/experiment/sft/dev.jsonl \
  --model Qwen/Qwen2.5-Coder-7B-Instruct --revision "$MODEL_REVISION" \
  --output checkpoints/sft-v3-seed42 \
  --epochs 3 --lr 2e-5 --batch-size 1 --grad-accum 8 \
  --max-length 2048 --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
  --patience 2 --min-delta 0.001 --seed 42 --dry-run
```

`--dry-run` 使用真实 tokenizer，但不加载模型或开始 GPU 更新。确认监督片段只包含目标 assistant 与结束 token，目标未被截断，train/dev 来源互斥。完整模板无法构造一致 mask 时应拒绝样本，不能靠字符长度估算 token 边界。

可先运行独立 tokenizer 审计，不下载模型权重：

```bash
python scripts/audit_tokenizer.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --revision c03e6d358207e414f1eca0bb1891e29f1db0e242 \
  --output results/tokenizer-audit.json
```

本次已在该 revision 上通过中文代码、多轮 assistant 历史和 Unicode/制表符三个案例。审计环境为 Transformers 5.22.1；这仅证明对应模板的 mask 行为，不能替代 `train` extra 版本组合的 GPU 验收。目标训练环境应重新运行审计。

完成检查后，使用同一命令去掉 `--dry-run`。默认单卡 `cuda:0`、NF4 double quant、bf16、SDPA、gradient checkpointing，packing 关闭。学习率比旧 `1e-4` 更保守，但最终选择由 dev 实测决定。

重点检查 `training-manifest.json`、`training-log.jsonl`、`best_adapter/` 和 `adapter/`：

- 实际 optimizer steps 大于零，损失和梯度有限，监督 token 大于零，LoRA 参数确实变化。
- 数据/配置/模板/实现哈希、resolved model revision、环境、seed、峰值显存与耗时完整。
- `adapter/` 是 dev loss 选择的结果；`best_epoch=0` 表示没有超过未训练基座，保存的是零增量 adapter。不能把基座被选中包装成 SFT 提升。
- 权重保存重载检查通过，且在新评测进程中加载 adapter。dev loss 降低仍需用 dev 执行成功率验证。

训练后的 `final_parameter_delta` 记录最后一轮实际更新，`parameter_delta` 记录最终选中的 adapter，两者不能混用。manifest 还包含监督 token 统计、初始/最终/选中参数哈希及峰值显存。CPU/tiny 模型回归、真实 tokenizer 审计和 GPU 训练验收是不同证据，测试通过不能替代训练收益。

默认保存 `epoch-001/adapter` 等 checkpoint 和旁置 manifest。训练后运行执行选择，Base 参与比较，平局保留 Base：

```bash
python scripts/select_checkpoint.py \
  --checkpoint-root checkpoints/sft-v3-seed42 \
  --tasks-file data/experiment/tasks.jsonl \
  --output results/sft-v3-dev-selection
```

后续使用 `selection.json` 的 `selected_adapter`，不要直接假定根目录 `adapter/` 是执行最优模型；值为 null 表示保留 Base。先用可审计小集合完成短训练，再扩大任务数量。前 1–2 个 epoch 已使 dev 执行能力下降时，检查样本质量、监督内容、任务覆盖和学习率；不要通过反复查看 test 选择新配方。

## 6. 公平评测与误差分析

正式 test 前先在 dev 上运行本节流程选择配方。test 只用于方案冻结后的确认；调整模型、prompt、token budget、量化或 task set 后，必须把相同调整同时应用于基座。

```bash
python scripts/evaluate.py \
  --tasks-file data/experiment/tasks.jsonl --split test \
  --model Qwen/Qwen2.5-Coder-7B-Instruct --revision "$MODEL_REVISION" \
  --training-data checkpoints/sft-v3-seed42/training-manifest.json \
  --quantization nf4 --temperature 0 --seed 42 --max-new-tokens 2048 \
  --output results/base-test.json
python scripts/evaluate.py \
  --tasks-file data/experiment/tasks.jsonl --split test \
  --model Qwen/Qwen2.5-Coder-7B-Instruct --revision "$MODEL_REVISION" \
  --adapter "$SELECTED_ADAPTER" \
  --training-data checkpoints/sft-v3-seed42/training-manifest.json \
  --quantization nf4 --temperature 0 --seed 42 --max-new-tokens 2048 \
  --output results/sft-test.json
python scripts/compare.py \
  --baseline results/base-test.json --treatment results/sft-test.json \
  --bootstrap-samples 10000 --seed 42 --output results/base-vs-sft-test.md
```

上述 `$SELECTED_ADAPTER` 必须先设为 dev 选择报告中的非空路径；选择 Base 时无需执行 adapter 对照，也不能声称 SFT 提升。`--training-data` 使用训练 manifest，使 test 同时检查 train 与用于选择的 dev 身份；两组使用同一审计输入。默认 greedy 的重复运行不会增加独立样本量。如果要研究采样，预先固定温度、每题候选数与 seed，单独命名为采样实验。

主指标为每题平均的 **pass@1**；多次中任意一次通过单独报告为 `any_pass_rate`，不能伪装成单次成功率。比较报告提供任务配对 bootstrap 差异区间及 any-pass Wilson 区间；只有每题单次生成时，后者也对应 pass@1 区间。完整实验还应统计改善/退化任务数、无效输出、截断、公开过而私有失败、基础设施错误、生成长度和耗时。重复生成与训练 seed 不能替代独立任务数量。

逐题查看完整生成、patch、grader details 与 `failure_type`。将格式失败、缺上下文、语义错误、边界回归、超时、环境故障分别分析。基础设施错误应修复后按同协议重跑，不能当成模型答错，也不能只删掉某一方的失败任务。小集合上置信区间很宽；同来源任务相关性仍需在技术报告中说明。

## 7. 实验矩阵与交付材料

优先完成少量能解释因果方向的对照，避免先堆 SFT / DPO / GRPO 名称：

| 实验 | 控制条件 | 要回答的问题 |
|---|---|---|
| Base → SFT | 同模型 revision、prompt、test、推理预算 | 新数据是否带来执行收益 |
| 学习率 `1e-5 / 2e-5 / 5e-5` | 相同 train/dev、seed 与 token 预算 | 是否存在更新过大或欠拟合 |
| 训练数据质量/覆盖消融 | 等监督 token，冻结 split | 收益来自任务覆盖还是重复样本 |
| 公开上下文消融 | 同任务与解码预算，独立重跑 Base/SFT | 依赖文件与公开测试贡献多少 |
| 成功答案 → 恢复轨迹 | 未来需补真实轨迹采集；等监督预算 | 失败反馈是否改善恢复能力 |
| SFT → 同状态 DPO | 未来需补真实 DPO trainer/reference | 偏好监督是否补充 SFT |

正式报告尽量使用 3 个训练 seed；选择过程只看 dev。训练次数、测试集规模、每次监督 token 和更新步数必须一起呈现。

建议准备以下复现包：

| 产物 | 内容 |
|---|---|
| 数据卡 | 来源与许可证、独立任务数、source_group 规则、去重/污染检查、split manifest |
| 训练记录 | 完整命令、环境锁文件、模型 commit、training manifest、loss 曲线、adapter 哈希 |
| 评测记录 | Base/SFT 原始 JSON、协议/任务哈希、逐题诊断、比较报告、置信区间 |
| 技术报告 | 问题定义、实现选择、消融、至少 3 个具体失败案例、局限 |
| 可演示材料 | 一个新任务从输入到修复、测试判分、训练后对照的演示；不要展示密钥或隐藏答案 |

2026-11-05 检查时 Git HEAD 可正常解析，原先对象损坏的记录已过时。实验需保存当前 commit、未提交 diff 和源码归档 SHA256，避免仅记录 commit 却遗漏本地改动。

