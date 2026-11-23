# 单卡 GPU 实验手册

[上手指南](START-HERE.md) · [实验成绩单](EXPERIMENT-RESULTS.md) · [训练与评测脚本](../scripts/)

从仓库根目录在同一个 Bash 会话中运行以下命令。可以先在 CPU 上完成任务审计与划分。教师采集会向已配置的服务发送公开任务并产生费用，QLoRA 训练与模型评测使用 GPU。报告中的 202 条训练样本来自一次历史采集，重新采样的条数和分数可能不同。

## 环境与模型

使用 Linux 或 WSL2、Python 3.10+ 和支持 bf16 的 CUDA GPU。报告使用 7B 模型、NF4 双量化、长度 2048 和 microbatch 1。24 GB 单卡可作起点，实际占用取决于序列长度、驱动和依赖版本。先安装与驱动匹配的 CUDA PyTorch，再安装训练依赖。[依赖范围](../pyproject.toml)

```bash
python -m pip install -e ".[dev,train,api]"
python -m pip check
python -m agentlab doctor
python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()'
```

### 判分用的沙箱

候选修复是要执行的模型生成代码。判分默认在内核 Landlock 沙箱里运行它：工作区只读，真实 HOME 与 `/proc` 不可读，TCP 连接被拒绝，并受 CPU 时间、地址空间、进程数的限制和进程组超时清理约束；环境变量按白名单构造。`agentlab doctor` 输出里的 `sandbox.*` 几行说明实际生效的隔离。

| `sandbox.landlock` | 判分行为 |
| --- | --- |
| available | 文件系统与 TCP 受内核限制 |
| unavailable | 退回到 rlimit 加清洁环境。文件系统与网络不受限，同一用户的 `/proc/<pid>/environ` 仍可读 |

`sandbox.selftest` 这一行是另一回事：它在沙箱里真实启动一次当前的解释器并导入判分用到的库。内核支持 Landlock，不代表这种安装方式的 Python 一定能在沙箱里启动，conda、独立构建和容器镜像里的 Python 各有自己的库目录。这一行必须是 `ok`。`env` 阶段运行同一个自检，失败就停，并给出解释器的报错原文。

以普通用户运行，root 默认被拒绝，`AGENTLAB_ALLOW_ROOT=1` 可以解除。在服务器上建议设置 `AGENTLAB_SANDBOX=required`：Landlock 不可用时直接停止，而不是降级。`AGENTLAB_SANDBOX=off` 显式关闭。

已知边界：另行 `setsid` 的孙进程不受进程组清理约束；Landlock 只覆盖 TCP；同一个 pytest 进程内的对抗性代码仍可能伪造输出。需要更强的边界时应使用容器，容器后端尚未实现。[隔离实现](../src/agentlab/runtime/isolation.py)

### 换用 Qwen3.5

Qwen3.5 没有 Coder 专版，与 7B 最接近的档位是 Qwen3.5-9B。它与 Qwen2.5 的差别都在代码里处理，切换只需要改模型名。

| 差别 | 仓库里的处理 |
| --- | --- |
| chat template 默认开启思考模式 | 所有调用统一走 `hf_runtime.apply_chat_template`，一律关闭。Qwen2.5 的模板没有这个开关，token 不变 |
| 每 4 层有 3 层是线性注意力，投影名不同 | LoRA 目标按模型里真实存在的模块选择 |
| 需要 transformers ≥ 5.2.0 | 依赖分出 `train-qwen35`；`env` 阶段在下载权重之前检查架构 |
| revision | 未指定时用该模型已核对的固定版本。换模型不会沿用旧模型的 revision |

```bash
MODEL=Qwen/Qwen3.5-9B bash scripts/run_experiment_v2.sh deps    # 安装 .[dev,train-qwen35,api]
MODEL=Qwen/Qwen3.5-9B bash scripts/run_experiment_v2.sh env
```

Qwen3.5 与 transformers 4.x 不兼容，建议在新的虚拟环境里安装。Qwen3.5-9B 的 BF16 权重约 18 GiB，NF4 约 8 GiB，24 GB 单卡可以完成 SFT 与评测；DPO 要同时放 reference 和 policy 两个模型，是最紧的一步。27B 和 35B-A3B 的 NF4 权重分别约 18 GiB 和 20 GiB，不适合在 24 GB 上做 QLoRA 训练。

**Qwen3.5 在 GPU 上的运行尚未验证。** NF4 量化、SFT、DPO 和评测目前只做过 CPU 层面的检查：tokenizer 审计、模型结构、加载参数。先做一次几条样本的冒烟测试，再铺开：

```bash
python scripts/audit_tokenizer.py --model Qwen/Qwen3.5-9B     # thinking_disabled 应为 true
python scripts/train_sft.py --data "$RUN/sft/train.jsonl" --eval-data "$RUN/sft/dev.jsonl" \
  --model Qwen/Qwen3.5-9B --output checkpoints/smoke-qwen35 --epochs 1 --dry-run
```

## 冻结任务池

只做这一节时，安装 `.[dev]` 即可。用一个新的实验目录导出 `pilot` 和四批扩展任务。导出器会运行参考修复与原始缺陷的检查，合并后的任务文件含有隐藏测试和参考答案，应保留在本地。

```bash
RUN=data/repair-104
(
  set -e
  mkdir -p "$RUN"
  mkdir "$RUN/pools"
  python scripts/export_pilot_tasks.py --output "$RUN/pools/pilot.jsonl"
  for pool in v2 v2b v2c v2d; do
    python scripts/export_tasks_v2.py --pool "$pool" --output "$RUN/pools/$pool.jsonl"
  done
  cat "$RUN"/pools/{pilot,v2,v2b,v2c,v2d}.jsonl > "$RUN/pool.jsonl"
  python scripts/prepare_data.py --split-tasks --tasks-file "$RUN/pool.jsonl" \
    --dev-ratio 0.20 --test-ratio 0.20 --seed 42 --output-dir "$RUN/frozen"
)
```

`frozen/split-manifest.json` 记录划分与任务哈希，`frozen/audit-manifest.json` 记录参考修复通过数和缺陷复现数。104 题共有 52 个来源组，每组两题。比例作用于**组数**并向下取整。

| 设置 | train | dev | test |
| --- | ---: | ---: | ---: |
| 本例显式设置 0.20 / 0.20 | 64 | 20 | 20 |
| 脚本默认 0.15 / 0.15 | 76 | 14 | 14 |

替换为自己的任务时，JSONL 每行描述一道修复题。`instruction`、`module`、`buggy`、`public_test` 和可选的 `extra_files` 构成公开上下文。`fixed` 和 `private_tests` 只交给判分器。同一仓库、模板及其近重复变体应使用相同的 `source_group`。当前输出协议允许替换一个主文件。[任务格式校验](../src/agentlab/benchmark/protocol.py)

需审计测试强度时，运行 `python scripts/audit_mutations.py --help` 查看变异审计参数。某个变异体通过所有测试，表示这组测试未区分这一处修改，值得复核该任务。

## 采集并重放教师答案

在本地会话或密钥管理器中设置 `TEACHER_BASE_URL`、`TEACHER_MODEL`、`TEACHER_API_KEY`。脚本内置的服务地址和模型名是占位值。模型只应收到公开指令、缺陷代码、必要依赖及公开测试。采集记录和 manifest 可能包含服务地址、生成内容和判分细节。

先查看 train 的候选槽位。`--status-only` 无需密钥，也不会发送请求。退出码 `0` 表示已完成，`3` 表示尚有待采集或可重试的槽位。

```bash
TASKS="$RUN/frozen/tasks.jsonl"
python scripts/collect_data.py --tasks-file "$TASKS" --split train \
  --candidates 4 --output "$RUN/train-candidates.jsonl" --status-only
```

确认服务和预算后，再执行下面的采集。按本例的划分与候选数，空目录会有 256 个 train 槽位与 40 个 dev 槽位。test 保留给终评。

```bash
python scripts/collect_data.py --tasks-file "$TASKS" --split train \
  --candidates 4 --output "$RUN/train-candidates.jsonl"
python scripts/collect_data.py --tasks-file "$TASKS" --split dev \
  --candidates 2 --output "$RUN/dev-candidates.jsonl"
python scripts/prepare_data.py --tasks-file "$TASKS" \
  --input "$RUN/train-candidates.jsonl" "$RUN/dev-candidates.jsonl" \
  --max-per-task 4 --output-dir "$RUN/sft"
```

数据准备阶段会去重、重放候选并验证通过标签。查看 `sft/dataset-manifest.json` 中的题目覆盖、拒绝原因与实际样本数。`--max-per-task 4` 是每题上限，默认值为 1。训练集里同一道题的多个解只代表一道独立任务。

## 训练与 dev 选型

固定模型和 tokenizer 的 revision。下面使用仓库记录的公开模型版本。每次训练使用新的 checkpoint 和结果目录。

```bash
MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
REVISION=c03e6d358207e414f1eca0bb1891e29f1db0e242
CKPT=checkpoints/repair-104-seed42
RESULTS=results/repair-104-seed42

train_args=(
  --data "$RUN/sft/train.jsonl" --eval-data "$RUN/sft/dev.jsonl"
  --model "$MODEL" --revision "$REVISION" --output "$CKPT"
  --epochs 3 --lr 2e-5 --batch-size 1 --grad-accum 8
  --max-length 2048 --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --seed 42
)
python scripts/train_sft.py "${train_args[@]}" --dry-run
```

`--dry-run` 加载真实 tokenizer 来检查数据和监督范围。确认有效监督 token 和截断统计后，再执行训练与 checkpoint 选择。

```bash
python scripts/train_sft.py "${train_args[@]}"
python scripts/select_checkpoint.py --checkpoint-root "$CKPT" \
  --tasks-file "$TASKS" --output "$RESULTS/dev" \
  --quantization nf4 --max-new-tokens 2048 --seed 42
```

| 在哪里看 | 回答什么问题 |
| --- | --- |
| `$CKPT/training-manifest.json` | 使用了哪版模型和哪些数据？实际更新了几步？ |
| `$CKPT/checkpoint-index.json` | 各 epoch 的 adapter 在哪里，文件哈希是什么？ |
| `$RESULTS/dev/selection.json` | 哪个 checkpoint 的 dev 执行通过率最高？`selected_adapter` 为 `null` 时保留 Base。 |

训练目录根部的 `adapter/` 按 dev token loss 保存。终评使用 `selection.json` 的执行选型结果。

## 冻结配置后评测 test

Base 和选中模型共享同一组任务、模型版本、训练数据清单及 greedy 解码设置。读取 dev 的选择结果后，再决定是否需要加载 adapter。

```bash
(
  set -e
  adapter=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["selected_adapter"] or "")' \
    "$RESULTS/dev/selection.json")
  eval_args=(
    --tasks-file "$TASKS" --split test --model "$MODEL" --revision "$REVISION"
    --training-data "$CKPT/training-manifest.json" --quantization nf4
    --temperature 0 --seed 42 --max-new-tokens 2048
  )
  python scripts/evaluate.py "${eval_args[@]}" --output "$RESULTS/base-test.json"
  if [[ -n "$adapter" ]]; then
    python scripts/evaluate.py "${eval_args[@]}" --adapter "$adapter" \
      --output "$RESULTS/selected-test.json"
    python scripts/compare.py --baseline "$RESULTS/base-test.json" \
      --treatment "$RESULTS/selected-test.json" --bootstrap-samples 10000 \
      --seed 42 --output "$RESULTS/comparison.md"
  else
    echo 'dev 选中了 Base，本次 test 相对 Base 的差值为 0。'
  fi
)
```

`pass@1` 在本例中是一次生成通过全部测试的任务比例。比较报告还统计逐题改善、退化和配对差值区间。差值区间默认做聚类 bootstrap，以来源组为抽样单位，与划分和泄漏检查使用同一个独立单位。`compare.py --cluster-by task` 与 `aggregate_seeds.py --cluster-by task` 复现按题重采样的历史口径。多次生成中任意一次通过是另一项指标 `any_pass_rate`。多训练种子应沿用同一份冻结划分和相同的 test 题目。[跨种子汇总脚本](../scripts/aggregate_seeds.py)

## 使用阶段脚本或分享报告

[阶段脚本](../scripts/run_experiment_v2.sh)提供 `pools`、`split`、`collect`、`prepare`、`train`、`select`、`eval` 等入口。`all` 会安装依赖并在缺少候选时进入付费采集，默认流程止于 dev。锁定配置后才单独调用 `final-eval`。阶段脚本有自己的目录、缓存和默认划分，以上手动示例在自己的目录中完成。

对外分享时使用汇总得分、划分规模、公开模型 revision 和方法说明。`data/`、`results/`、`checkpoints/`、`.state/` 与原始终端日志保留在本地，尤其留意 manifest 中的服务地址和绝对路径。[发布前检查](START-HERE.md#分享实验结果)

## 导出可公开的证据

终评完成后，把可以公开的部分整理进仓库。导出只保留判定与计数，不带走模型输出、私有测试细节、服务地址、绝对路径、硬件型号和 API key，并先按 `privacy_check.py` 的规则自查。命中任何一条都不会写出文件。

```bash
python scripts/export_evidence.py --label repair-104-seed42 \
  --results-dir "$RESULTS" --checkpoint-dir "$CKPT" \
  --candidates-manifest data/repair-104/train-candidates.manifest.json
python scripts/verify_evidence.py docs/evidence/repair-104-seed42
python scripts/privacy_check.py
```

`--candidates-manifest` 可选，它记录教师模型名；服务地址会被替换掉。

`verify_evidence.py` 从逐题结果重算通过数、dev 选型和 bootstrap 区间，并核对 `selection.json` 与 `comparison.md`。README 引用了哪个数字，就把它写进证据包里的 `claims.json`，例如 `{"base_test_passed": 12, "selected_test_passed": 13}`。复核脚本逐项比对，CI 对每个已提交的证据包重复这一步。
