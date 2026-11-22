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

`pass@1` 在本例中是一次生成通过全部测试的任务比例。比较报告还统计逐题改善、退化和配对差值区间。多次生成中任意一次通过是另一项指标 `any_pass_rate`。多训练种子应沿用同一份冻结划分和相同的 test 题目。[跨种子汇总脚本](../scripts/aggregate_seeds.py)

## 使用阶段脚本或分享报告

[阶段脚本](../scripts/run_experiment_v2.sh)提供 `pools`、`split`、`collect`、`prepare`、`train`、`select`、`eval` 等入口。`all` 会安装依赖并在缺少候选时进入付费采集，默认流程止于 dev。锁定配置后才单独调用 `final-eval`。阶段脚本有自己的目录、缓存和默认划分，以上手动示例在自己的目录中完成。

对外分享时使用汇总得分、划分规模、公开模型 revision 和方法说明。`data/`、`results/`、`checkpoints/`、`.state/` 与原始终端日志保留在本地，尤其留意 manifest 中的服务地址和绝对路径。[发布前检查](START-HERE.md#分享实验结果)
