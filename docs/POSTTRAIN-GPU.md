# 单卡 GPU 实验手册

[上手指南](START-HERE.md) · [实验成绩单](EXPERIMENT-RESULTS.md) · [训练与评测脚本](../scripts/)

从仓库根目录在同一个 Bash 会话中运行以下命令。可以先在 CPU 上完成任务审计与划分。教师采集会向已配置的服务发送公开任务并产生费用，QLoRA 训练与模型评测使用 GPU。报告中的 202 条训练样本来自一次历史采集，重新采样的条数和分数可能不同。

## 环境与模型

用 `git clone` 或解压发布包都可以，解压后的目录没有 `.git` 不影响任何阶段。使用 Linux 或 WSL2、Python 3.10+ 和支持 bf16 的 CUDA GPU。已发表的报告使用 Qwen2.5 7B 模型、NF4 双量化、长度 2048 和 microbatch 1。24 GB 单卡可作起点，实际占用取决于序列长度、驱动和依赖版本。先安装与驱动匹配的 CUDA PyTorch，再安装训练依赖。[依赖范围](../pyproject.toml)

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

以普通用户运行，root 默认被拒绝，`AGENTLAB_ALLOW_ROOT=1` 可以解除。在服务器上建议设置 `AGENTLAB_SANDBOX=required`：Landlock 不可用时直接停止，而不是降级。`AGENTLAB_SANDBOX=off` 显式关闭。内核不支持 Landlock 的机器上不要设 `required`（设了会在 `env` 阶段直接停住）；本页后面的阶段脚本运行时都会把它取消掉，降级时会在日志里说明一次。

已知边界：另行 `setsid` 的孙进程不受进程组清理约束；Landlock 只覆盖 TCP；同一个 pytest 进程内的对抗性代码仍可能伪造输出。需要更强的边界时应使用容器，容器后端尚未实现。[隔离实现](../src/agentlab/runtime/isolation.py)

### 模型

默认模型是 Qwen3.5-9B，需要 transformers 5.2.0 以上；`.[dev,train,api]` 装的就是这套依赖。它的实验结果见 README 里「Qwen3.5-9B 上的结果」一节；NF4 量化、显存占用和输出质量只能在目标机器上验证。所以正式运行前先做冒烟测试，首次运行会下载约 18GB 权重，之后一两分钟内给出结果：

```bash
bash scripts/run_experiment_v2.sh smoke
```

它走与正式训练、评测相同的代码路径：加载模型、挂 LoRA、做一步训练、生成一次。报告里有每一步的耗时、峰值显存，以及生成的内容能否提取出可解析的 Python。任何一项失败都会以非零状态退出。`all` 会在付费采集之前自动运行它。

Qwen3.5 与 Qwen2.5 的差别都在代码里处理：

| 差别 | 仓库里的处理 |
| --- | --- |
| chat template 默认开启思考模式 | 所有调用统一走 `hf_runtime.apply_chat_template`，一律关闭；评测可以用 `--thinking` 打开（见下面的"思考模式"）。Qwen2.5 的模板没有这个开关，token 不变 |
| 每 4 层有 3 层是线性注意力，投影名不同 | LoRA 目标按模型里真实存在的模块选择 |
| 需要 transformers ≥ 5.2.0 | `env` 阶段在下载权重之前检查架构 |
| revision | 未指定时用该模型已核对的固定版本。换模型不会沿用旧模型的 revision |

线性注意力层有两个可选的加速内核，`causal_conv1d` 和 `flash-linear-attention`。不装也能运行，结果正确，但 transformers 会退回慢速的参考实现。冒烟报告的 `fast_kernels` 列出它们是否已安装，`tokens_per_second` 反映实际速度：

```bash
python -m pip install causal-conv1d flash-linear-attention
```

Qwen3.5-9B 的 BF16 权重约 18 GiB，NF4 约 8 GiB，24 GB 单卡可以完成 SFT 与评测；DPO 要同时放 reference 和 policy 两个模型，显存最紧，尚未验证。27B 和 35B-A3B 的 NF4 权重分别约 18 GiB 和 20 GiB，不适合在 24 GB 上做 QLoRA 训练。

### 复现已发表的结果：Qwen2.5

README 里的数字全部来自 Qwen2.5-Coder-7B-Instruct，它使用 transformers 4.x。复现时显式设置模型，阶段脚本会改用 `train-qwen25` 这套依赖：

```bash
MODEL=Qwen/Qwen2.5-Coder-7B-Instruct bash scripts/run_experiment_v2.sh deps
MODEL=Qwen/Qwen2.5-Coder-7B-Instruct bash scripts/run_experiment_v2.sh all
```

两种模型的 transformers 版本范围不重叠，需要各自的环境。

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

### 可选：追加难题池 `h3`

默认的 104 题对较强的基座已接近上限。难题池 `h3` 追加 40 道多缺陷题，共 20 个来源组，每族两道，分五个方向：状态一致性、事务与回滚、异常安全、锁序与原子性、条件变量与工作线程。每道题的有问题代码带有多个互相牵连的缺陷，只修最明显的那一处仍然过不了隐藏测试。题面用编号写明每一条要求，公开测试只展示接口，隐藏测试逐条覆盖。

题池是可选项。默认基准和已冻结的实验目录都不变：

```bash
EXTRA_POOLS=h3 bash scripts/run_experiment_v2.sh pools    # 合并后 144 题
EXTRA_POOLS=h3 bash scripts/run_experiment_v2.sh split
```

`EXTRA_POOLS` 进入运行指纹，启用它的实验有自己的状态目录，之后的每个阶段都要带上这个变量。

启用可选池时，划分会按层进行：带 `stratum` 字段的来源组（难题池）按比例进入三个集合，而不是听凭一次打乱。144 题按 0.20 / 0.20 划分时，40 道难题是 train 24、dev 8、test 8，原有 104 题仍是 64 / 20 / 20；不分层时难题在 test 里只有 4 道。`split-manifest.json` 的 `stratified` 一节记录每层分到的组数。默认题池只有一层，划分与以往完全相同。

**难度阶梯 `h3s`。** 每道难题另有一组单缺陷变体：题面、参考解和测试都不变，程序里只留一处缺陷。它是难题和原有题之间的一级，用 `EXTRA_POOLS="h3 h3s"` 启用。变体与母题同属一个来源组，永远落在同一个划分里，所以 test 中难题来源族的题会成倍增加。导出时会重放每个变体，确认参考解通过、变体过公开测试而挂隐藏测试。基座在变体上的通过率需要先测，方法与上面测整池相同。

训练之前先量基座在整池上的通过率，题目够不够难以这个数为准：

```bash
RUN=data/repair-h3
MODEL=Qwen/Qwen3.5-9B
REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a
python scripts/export_tasks_v2.py --pool h3 --output "$RUN/pool.jsonl"
python - "$RUN" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
specs = [json.loads(line) for line in (run / "pool.jsonl").read_text(encoding="utf-8").splitlines()
         if line.strip()]
with (run / "tasks-ood.jsonl").open("w", encoding="utf-8") as handle:
    for spec in specs:
        handle.write(json.dumps({**spec, "split": "ood"}, ensure_ascii=False, sort_keys=True) + "\n")
PY
python scripts/evaluate.py --tasks-file "$RUN/tasks-ood.jsonl" --split ood \
  --model "$MODEL" --revision "$REVISION" --quantization nf4 --temperature 0 \
  --seed 42 --max-new-tokens 2048 --output "$RUN/results/base-ood.json"
```

`split=ood` 把整池当作留出集。导出时会重放每道题的参考解和原始程序，早于任何划分和训练。

三点注意：

- **长度。** 超过 `MAX_LENGTH`（默认 2048）的训练样本会被跳过而不是截断，训练记录里的 `skipped_long` 应当为 0。`python scripts/self_check_pool.py --module agentlab.benchmark.hard_tasks_v3 --attr HARD3_SPECS --no-grade` 打印每道题的 token 粗估。
- **耗时。** 难题的答案更长，评测和采样都比默认题池慢得多。教师采集不受影响，先走 SFT。
- **加题或加难度时。** 新题用 `hard_tasks_v3_common._task` 构造，并为每条要求配一个"部分修复"变体。`python scripts/self_check_pool.py --module <题目模块>` 会重放参考解、原始程序和全部变体，任何一项不符合都以非零状态退出。

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

损失是分段算的（`agentlab.training.sequence_loss`）：主干只过一次，lm_head 和交叉熵只对被监督的 token、每 256 个一段地计算，每一段在反向时重算。模型自带的做法会先放下整段序列的全部 logits：词表 24.8 万时，2048 个 token 的一份拷贝就是 2 GiB，几份同时存在，24 GB 的卡在长样本上会显存不足。分段算的结果与自带的损失相同（差别在浮点误差内），训练开始前会在最短样本的一段上核对两者，不一致就停下；`--no-chunked-loss` 换回自带的做法。

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

### 思考模式

Qwen3.5 默认先在 `<think>` 里推理、再给答案；本项目默认关闭思考（`enable_thinking=False`），回答就是模型输出的全部内容，上面的协议哈希和已有结果都基于这一点。要量一下思考模式的效果，在评测脚本上加 `--thinking`，并按模型卡的推荐用采样而不是贪心：

```bash
python scripts/evaluate.py --tasks-file "$TASKS" --split ood --model "$MODEL" --revision "$REVISION" \
  --quantization nf4 --temperature 0.6 --top-p 0.95 --top-k 20 --thinking \
  --max-new-tokens 8192 --output "$RESULTS/thinking.json"
```

`--top-p` 和 `--top-k` 只在 `--temperature` 大于 0 时生效，取非默认值（1.0 和 0）时才写进协议，以往的采样评测不受影响。打开思考后只对 `</think>` 之后的回答判分；生成额度用完时推理还没结束的回答记为 `thinking_unfinished`，不进判分器，也不算代码错误。推理会计入输出 token，所以 `--max-new-tokens` 要给够（官方典型输出长度是 32,768 token），否则会把“没想完”误当成“做不出”。协议里会多出 `generation.thinking: true`，与关闭思考的报告不能直接配对比较；`--thinking` 目前只支持单次生成和本地模型。阶段脚本用 `THINKING=1` 打开思考，仍是贪心解码；正式协议要不要改成采样，以实测为准。

先小规模量一下值不值得切换协议：`scripts/thinking_pilot.sh` 从一份冻结划分（放在 `data/frozen-ladder.jsonl`）的 train 和 dev 题里选 12 道难题和 12 道单缺陷变体，按三种解码设置各评一次并逐题对比，不碰 test 题。

| 组 | 设置 |
| --- | --- |
| A | 现有协议：思考关闭，贪心 |
| B | 采样、不思考：temperature 0.7 / top_p 0.8 / top_k 20 |
| C | 采样、思考：temperature 0.6 / top_p 0.95 / top_k 20，上限默认 8192 token（`PILOT_MAX` 可调） |

A、B 两组的上限是 2048 token。未装加速内核时 C 组每道题最长十来分钟，整份实测要几个小时。某一组的结果文件已经存在、且不是整份环境报错时，这一组跳过，中断后重新启动会接着跑。`bash scripts/show_progress.sh <日志文件>` 打印一次进度面板，前面加 `WATCH=30` 则每 30 秒刷新一次、任务结束后自动退出。

## 生成提速：vLLM

用 transformers 逐条生成只有二十来 token/s，采样几千次要十几个小时。vLLM 服务并发请求时合计有几百 token/s（Qwen3.5-9B，单卡 24 GB，bf16）。它要占几乎整块显卡，所以不能和评测、训练同时跑：采样时启动它，采完就关。

vLLM 会替换 torch 等依赖，必须装在独立的环境里，现有的训练环境不动：

```bash
python -m venv ~/vllm-qwen
~/vllm-qwen/bin/python -m pip install -U "vllm>=0.27"
HF_ENDPOINT=https://hf-mirror.com ~/vllm-qwen/bin/vllm serve Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --language-model-only \
  --max-model-len 8192 --port 8000
```

等日志出现 `Application startup complete` 后，在另一个终端测速，然后关掉服务：

```bash
python scripts/vllm_speed.py        # 单路 3 次 + 16 路并发
pkill -f 'vllm serve'
```

显存不够时服务起不来：先确认显卡是空的（`nvidia-smi`）。环境里设了 `http_proxy` 之类的变量时，访问本机服务的程序要绕开代理，仓库里的客户端（`agentlab.sampling.vllm_client`）已经这样做了。

## RFT：让模型自己采样，只用判分通过的答案训练

教师答案的 SFT 没有帮过这个基座，模型自己采出来、并且通过隐藏测试的答案才有帮助。`scripts/run_rft.sh` 把一轮做完：

1. 启动 vLLM 并等它就绪。
2. `scripts/sample_rft.py` 对 train 和 dev 的难题、单缺陷变体各采样 8 次（偶数号 0.7、奇数号 1.0，top_p 0.95，关闭思考），每个样本都用隐藏测试判分；采完关掉 vLLM。
3. `prepare_data.py` 重放验证通过的样本，每题留一条；`scripts/rft_control.py` 构造对照组。
4. 三组各训练 1 个 epoch，都在 dev 上选型：

   | 组 | 训练数据 | 学习率 / LoRA 秩 |
   | --- | --- | --- |
   | A 验证组 | 只留判分通过的答案 | 1e-4 / 32 |
   | C 对照组 | 同样的题，每题随机一份不检查对错的答案 | 1e-4 / 32 |
   | B 旧超参 | 同 A | 1e-5 / 16 |

5. `scripts/tier_report.py` 按 原有题 / 难题 / 变体 汇总。

对照组检验的是“按判分结果筛选”这一步有没有用：如果 C 在 dev 上的变化和 A 一样，提升就不是验证带来的。test 这一轮不评；看完 dev 的结果、定下评哪一组之后，用 `scripts/rft_test.sh <组名>` 在 test 上评一次（基座和选中的模型各一遍，约一小时），得到 `comparison.md` 和按层的汇总。它读那一组的选型结果，保留基座时什么都不评；已经评完的报告会保留。整轮大约三四个小时，大头是每个模型在 66 道 dev 题上的评测（每个约半小时）。

```bash
nohup bash scripts/run_rft.sh > rft-round1.log 2>&1 &
WATCH=30 bash scripts/show_progress.sh rft-round1.log "" run_rft.sh
```

每一步的产物都在就跳过，中断后重新启动会接着跑。采样目录（`data/rft-round1/sampling`）里有：

| 文件 | 内容 |
| --- | --- |
| `<split>-raw.jsonl` | 模型的原始输出，一条一次采样 |
| `<split>-graded.jsonl` | 全部样本及判分结果 |
| `<split>-candidates.jsonl` | 判分通过的，格式与教师采集相同 |
| `<split>-manifest.json` | 每层每题的通过次数、pass@k、失败类型 |

一次都没通过的题列在 manifest 的 `zero_pass_tasks` 里。常用变量：`RFT_SAMPLES`（每题采样次数）、`RFT_TIERS`（采样的层）、`RFT_LR` / `RFT_LORA_R`（A、C 组）、`OLD_LR` / `OLD_LORA_R`（B 组）、`RFT_EPOCHS`、`VLLM_BIN`（vLLM 的路径，默认 `~/vllm-qwen/bin/vllm`）、`TASKS_FILE`（冻结的划分，默认 `data/frozen-ladder.jsonl`）。

`select_checkpoint.py` 有两个省时间的开关：`--base-report` 复用已有的基座 dev 报告（题目、模型版本、生成设置都一致才接受），`--resume` 在中断后接着评，已经评完的 epoch 不重评。

`run_rft.sh` 另有三个开关，都是环境变量：

| 开关 | 作用 |
| --- | --- |
| `SKIP_ARMS="B-oldhp"` | 空格分隔的组名，这些组不训；最后的汇总和 vLLM 评测也只列没跳过的组 |
| `STAGES=sampling` | 采样做完就关掉 vLLM 并停下，训练另找时间做（题多到一晚跑不完时拆成两晚）。它不打印"全部结束"，所以停下以后进度面板显示的是"状态: 没有在运行（中途中断了，看最后几行找原因）"，第 2 步还标着"进行中"，看着像出了故障：日志的倒数第二行是 `结果：采样阶段结束`、最后一行是它后面那句括号提示（`（STAGES=sampling：到这里就停。…）`），就是正常停下。第二次去掉这个变量再运行，采样已经做完，会直接跳到准备数据 |
| `VLLM_EVAL=1` | 汇总之后经本机 vLLM 服务把没跳过的组再评一遍（辅助指标，见下面的"vLLM 评测"一节）。没评完时最后一行是"没有全部完成"，不影响 dev 上的选型结果 |

**题池扩大以后。** 合成出题（见"合成出题"一节）把新题并进 `data/frozen-ladder-v2.jsonl` 以后，在这份阶梯上重做第 1 轮要换任务文件、换运行目录、多采两层：

```bash
TASKS_FILE=data/frozen-ladder-v2.jsonl RUN_DIR=data/rft-expanded RFT_TIERS=hard,variant,synth_hard,synth_variant \
  nohup bash scripts/run_rft.sh > rft-expanded.log 2>&1 &
WATCH=30 bash scripts/show_progress.sh rft-expanded.log "" run_rft.sh
```

`RUN_DIR` 要给一个新的目录：采样协议里记着任务文件和采哪几层，和 `data/rft-round1` 里已有的对不上时，`sample_rft.py` 以退出码 2 结束。和第 1 轮比有三处不同：第 1 步多一行 `其中合成题（s3_、s3v_ 开头）train … / dev … / test …`；采样汇总和最后的表多出"合成难题""合成变体"两栏（第 2 步的标记里仍写着"难题和变体"，那是写死的）；题多了以后采样、各组的训练和选型都跟着变慢，一晚跑不完就用上面的 `STAGES=sampling` 拆成两晚。这一轮具体怎么安排要等出题的结果再定：先看每批接受了多少族、基座在合成题上能做对多少，再定采哪几层、要不要拆。

### 精度对照：在 bf16 或 int8 下重训，再经 vLLM 评测

为什么要做：RFT 那一轮是在 NF4 下训练、NF4 下选型的。同一组适配器放到 bf16 的 vLLM 里再评一遍，NF4 下的提升就没有了，基座自己还比 NF4 下多通过 16 道（见 README 的"两套评测的对照"）。`scripts/rft_precision_probe.sh` 用那一轮现成的训练数据、同样的超参，只换训练的精度，把各组的适配器重训一遍，再交给 vLLM 评测，回答的是：训练本身放到和部署一样的精度里，提升还在不在。

```bash
cd ~/jupyterlab/CodeAgent-R-v24 && if pgrep -f 'rft_precision_probe[.]sh' > /dev/null; then echo "已经在运行，不重复启动"; else TASKS_FILE=data/frozen-ladder-v2.1127.jsonl nohup bash scripts/rft_precision_probe.sh > rft-precision-bf16.log 2>&1 < /dev/null & echo "已启动"; fi
```

贴完应该看到 `已启动`。默认 `QUANT=bf16`，数据取自 `data/rft-expanded-1127`，重训 A-verified 和 C-control 两组：每组先训练，再评基座和两组（dev 165 题，贪心加每题采样 4 次）。估 40–60 分钟。bf16 的 9B 模型光权重就约 18 GB，能不能放下只有上了显卡才知道：训练报显存不足时脚本会说明，换成 int8 再跑一次（8 位权重，显存约一半，精度比 NF4 更接近 bf16）：

```bash
cd ~/jupyterlab/CodeAgent-R-v24 && if pgrep -f 'rft_precision_probe[.]sh' > /dev/null; then echo "已经在运行，不重复启动"; else TASKS_FILE=data/frozen-ladder-v2.1127.jsonl QUANT=int8 nohup bash scripts/rft_precision_probe.sh > rft-precision-int8.log 2>&1 < /dev/null & echo "已启动"; fi
```

看进度、取结果（int8 那一跑把日志名换成 `rft-precision-int8.log`）：

```bash
cd ~/jupyterlab/CodeAgent-R-v24 && WATCH=30 bash scripts/show_progress.sh rft-precision-bf16.log "" rft_precision_probe.sh
```
```bash
grep -aE '^===== |结果：|显存不够|^(标签|base|A-verified|C-control)' ~/jupyterlab/CodeAgent-R-v24/rft-precision-bf16.log | python3 -c 'import sys; print("\n".join(l[:240] for l in sys.stdin.buffer.read().decode("utf-8", "replace").splitlines()))'
```

读法：和 README「两套评测的对照」那张表放在一起看。NF4 下训练的适配器在 vLLM 下是基座 84、A-verified 85、C-control 76。换成 bf16 或 int8 训练以后，A-verified 如果明显高于基座和对照组，说明之前的"没有提升"是训练精度造成的；如果还是一样，就是在更接近部署的精度下又得到了一次"答案对错不重要"。常用变量：`SOURCE_DIR`（取数据的那一轮的运行目录）、`TASKS_FILE`（要和那一轮用的是同一份阶梯，用它的快照最稳）、`ARMS`、`EPOCHS`、`LR`、`EVAL_SAMPLES`。中断后原样重跑，训练完的组不重训，vLLM 评过的部分跳过。这一跑不做 NF4 下的 dev 选型，直接拿最后一个 epoch 去评。

## 引导采样的一轮：前缀提示与教师技能

第 1 轮在 66 道 dev 题上没有令人信服的提升：验证组和不验证的对照组打平，那点变化不是"验证"带来的。原因有三个：训练数据只有 61 条，1 个 epoch 只有 8 步更新；train 有 72 道题（22 道难题、50 道变体）基座采 8 次一次都没对，验证组里没有它们的正确答案；dev 只有 66 题、又只看一次贪心，上下两三道题的变化分不出来。这一轮带着提示把那 72 道题再采一遍，给前两条补信号，第三条靠下一节的辅助指标。第 1 轮要先跑完，产物在 `data/rft-round1`；`scripts/run_rft_guided.sh` 一次做完，默认 13 步（进度面板里的编号就是它）：

1. 检查前提：任务文件、第 1 轮的产物、vLLM 的路径、显卡是否空闲。
2. 写探针题号文件：上一轮零通过的题里属于难题、变体两层的，之后当族内探针用。
3. 启动 vLLM，先经 vLLM 评一遍基座（辅助指标，见下一节）。
4. 教师给零通过的题写技能提示（`scripts/write_skills.py`）。
5. 前缀引导采样。
6. 技能引导采样。
7. 关闭 vLLM，等显存释放。
8. 准备训练数据：两份引导组，和逐题配对的对照。
9–11. D-prefix、D-skill、F-control 各训练 1 个 epoch，都在 dev 上选型。
12. 汇总主结果：HF 贪心表，第 1 轮的两组和这一轮的三组放在一起。
13. 经 vLLM 评各组选中的适配器，打印两张表和读法。

第 5、6 步是 `scripts/sample_rft.py` 的提示模式，每个样本都用隐藏测试判分，只留通过的。可用的技能太少（不到 `MIN_SKILL_TASKS`，默认 5 道）时没有技能引导采样和 D-skill 这两步：总步数变成 11，但这个决定是第 4 步做完才下的，所以日志里是 `1/13` 到 `4/13` 接着 `5/11` 到 `11/11`。`SKILL=0` 时连写技能也没有，一共 10 步，从头就是 `n/10`；`VLLM_EVAL=0` 时少一步，日志末尾也只有一行"全部结束"。

提示只加在采样时发出去的用户消息末尾；训练记录里的 `messages` 仍是不带提示的标准提示，训练时模型看不到它。采样参数和第 1 轮一样，只采上一轮零通过的题：

| 提示 | 做法 |
| --- | --- |
| 前缀提示 | 把参考修复主文件的开头按 25%、50%、75% 三种比例放进用户消息（至少留一行让模型自己写），要求给出的开头原样保留、接着写完；每种比例各采 4 次。每道题只把最短的、成功过的比例交给训练（arXiv:2601.18779 的取法），更长比例采到的答案留在 `graded` 文件里 |
| 技能提示 | 教师看题目、有缺陷的文件、参考修复的 diff 和最多两份基座的失败尝试，写一条四段的通用技能（适用条件、要点、常见错误、推荐做法），不出现代码和这道题里的名字；每题采 8 次。每写一版都过两道检查：静态检查（不能出现参考修复和测试里的名字、字面量）和教师自判（只凭这条技能加题目，能否基本重建参考修复），没过就带着意见重写，最多 2 轮，仍没过就不用。泄露了答案的技能训出来的模型更差（arXiv:2608.08570） |

三组的超参相同（lr 1e-4、LoRA 秩 32、1 个 epoch），全部只在 dev 上选型，test 这一轮不评：

| 组 | 训练数据 |
| --- | --- |
| D-prefix | 第 1 轮的验证组 + 前缀提示引导出的正确答案 |
| D-skill | 第 1 轮的验证组 + 技能提示引导出的正确答案 |
| F-control | 第 1 轮的验证组 + 同样这些引导题各一份随机取的、判分没通过的答案（训练加 `--allow-unverified`） |

第 1 轮的对照组换成随机答案的题大多是基座本来就会做的，对错差别不大；这一轮的差别只落在"带着提示才做出来的题"上。F-control 的引导题是两个 D 组引导题的并集：各 D 组合起来和它逐题对得上，所以它比单独一个 D 组多出另一个 D 组独有的题。一道题在第 1 轮的采样里连一份写完了、格式完整、又没判分通过的答案都没有时（例如全是被截断的），每个 D 组都不留这道题，题号记在 `sft/F-control/unmatched-tasks.txt`。F-control 的答案取自第 1 轮不带提示的采样，和 D 组的正确答案相比，差别不只是对错，还有采样时有没有提示。

**读法。** 主结果是"汇总主结果"那一步打印的 HF 贪心表（原有题 / 难题 / 变体 / 合计）：第 1 轮的 A-verified、C-control 和这一轮的三组各占两行，`base` 是同一份基座报告，`epoch 1（选中）` 是这一组选中的模型；表下面是每个选中的模型一行"相对 …：变好 a 道、变差 b 道"（第一组之后，每组还各带一行 `base · <组> 相对 …：变好 0 道、变差 0 道`，是基座报告和自己比，恒为 0，不用看）。dev 只有 66 题，第 1 轮里上下两三道题的变化是训练扰动带来的随机效应。D 组要比 F-control 高出不止这个数，才说明难题的正确答案是有用的训练信号；两组差不多，就是在这个规模下训练答案的对错不重要。另有两行要一起看：`结果：D-prefix 训练集 N 条 / M 题，其中引导题 K 条` 说明 72 道里引导出了几道；`结果：训练时实际加载的题 … 逐题相同` 说明各 D 组合起来和 F-control 训练时用的是同一批题，不是的话会列出对不上的题号，多半是有记录超过 `MAX_LENGTH` 被训练入口丢掉了。

整轮约 5 小时，适合过夜。这是按各步的题数和第 1 轮测过的速度推的估算，这一轮还没有在服务器上跑过：大头是三组各在 66 道 dev 题上的贪心选型（每个约半小时）、经 vLLM 的四次评测（基座一次、三组各一次，共约 1600 个槽位）和两次引导采样（前缀 72 × 3 × 4 = 864 个槽位，技能最多 72 × 8 = 576 个，只算有可用技能的题）。写技能只调教师接口，不花显卡时间。

### 启动、看进度、取结果

先把 `CodeAgent-R-v24.tar.gz` 上传到 `~/jupyterlab/`。第 1 轮的产物在 `CodeAgent-R-v19` 的 `data` 里，教师的密钥在旧包的 `.env` 里。下面这一条命令解压、把 `data` 接过来、找到 `.env` 就拷一份，然后在后台启动；重复粘贴不会起第二个。哪个旧包里都没有 `.env` 时它不启动，所以不会再出现"没有密钥也跑了一轮"：

<!-- deliver:guided-launch -->
```bash
cd ~/jupyterlab && tar -xzf CodeAgent-R-v24.tar.gz && cd CodeAgent-R-v24 && { [[ -e data ]] || ln -s ../CodeAgent-R-v19/data data; } && { if [[ ! -e .env ]]; then src=""; for f in ../CodeAgent-R-*/.env; do [[ -f "$f" ]] && src="$f"; done; if [[ -z "$src" ]]; then echo "没有找到 .env（里面要有 TEACHER_API_KEY）：把它放到 ~/jupyterlab/CodeAgent-R-v24/ 再粘一次"; false; else cp "$src" .env; echo "已从 $src 拷来 .env"; fi; fi; } && if pgrep -f run_rft_guided.sh > /dev/null; then echo "已经在运行，不重复启动"; else nohup bash scripts/run_rft_guided.sh > rft-guided.log 2>&1 < /dev/null & echo "已启动"; fi
```

第一次贴是两行：`已从 ../CodeAgent-R-v21/.env 拷来 .env`（是哪个旧包就写哪个；多个旧包里都有 `.env` 时取目录名排在最后的那个，版本号位数相同时就是版本号最大的，`v19` 和 `v21` 里取 `v21`）和 `已启动`（前面可能多一行 `[1] 进程号`）。`CodeAgent-R-v24` 里已经有 `.env` 时只打 `已启动`。看到 `没有找到 .env（里面要有 TEACHER_API_KEY）：把它放到 ~/jupyterlab/CodeAgent-R-v24/ 再粘一次`，是什么都没启动：把 `.env` 放进这个目录再粘一次。脚本已经在跑时看到的是 `已经在运行，不重复启动`，正在跑的那份不受影响。这一句只说明有哪个进程的命令行里带着脚本名，有两种情况脚本其实没在跑，这时什么都没启动：另一个终端里的进度面板还没退出，等它自己退出（半分钟内）或在那里按 Ctrl+C 退掉，再粘一次；这一条不是直接贴进终端，而是被包了一层执行的，`bash -c '…'` 这一类会匹配到它自己，要把它单独贴进终端。脚本开头的注释里另有一行启动命令（用 flock 防重复），不要和这一条混用：用这一条启动以后再去粘那一行，会起第二个；反过来先用那一行、再粘这一条，只会说"已经在运行"。

看到了别的报错，就是解压或接数据那一步失败了，什么都没启动。看到了 `已启动`、面板却说"没有在运行"，就是脚本在第 1 步就停下了（显卡被占、缺第 1 轮的产物、找不到 vLLM 之类）：先看 `~/jupyterlab/CodeAgent-R-v24/rft-guided.log` 的开头和最后几行，原因和该怎么做都写在里面。每次启动都会重写这个日志，上一次的想留着就先拷走。这一轮的产物在 `data` 目录下的 `rft-guided` 里，而 `data` 接的是上一包的 `data`，所以从 v22 重跑会接着上一跑做。

启动一两分钟以后、睡觉以前再确认一件事：教师的密钥读到了没有。读不到时这一轮照常跑完，但没有 D-skill 组，而面板上看不出来（下面那条看进度的命令不筛这种提醒）。

```bash
grep -n '没有 TEACHER_API_KEY' ~/jupyterlab/CodeAgent-R-v24/rft-guided.log
```

贴完什么都没有、直接回到提示符，就是读到了。打印出 `提醒：环境里没有 TEACHER_API_KEY…` 这一行（行首带着行号）就是没读到：拷来的 `.env` 里没有这一项（哪里都找不到 `.env` 时启动命令自己就停了，走不到这一步）。把 `TEACHER_API_KEY`（连同 `TEACHER_BASE_URL`、`TEACHER_MODEL`）写进 `~/jupyterlab/CodeAgent-R-v24/.env`，等这一跑结束后原样重跑，就会补上 D-skill 组。提示找不到 `rft-guided.log`，是脚本没起来，回头看上一条命令的报错。

看进度（每 30 秒刷新一次，这一轮结束后自动回到提示符；Ctrl+C 只是停止观看，不影响后台任务。关掉网页也不影响后台任务，下次打开终端接着看）：

<!-- deliver:guided-progress -->
```bash
cd ~/jupyterlab/CodeAgent-R-v24 && WATCH=30 bash scripts/show_progress.sh rft-guided.log "" run_rft_guided.sh
```

贴完应该看到一块面板：`状态: 进行中`，下面依次是 `步骤`（每一步标着 `[完成]` 或 `[进行中]`）、`当前进度`（最后几行进度条，最多 4 行；采样、评测、写技能的带已用和剩余时间，等 vLLM 加载的只有已用时间，刚启动的头几分钟就是后者）、`已出的结果`（到这时为止的 `结果：` 行、表格行和出错信息，只留最后 14 行）和 `日志最后 3 行`。最后一步里嵌着 vLLM 评测自己的 8 步，所以步骤不止 13 行。状态变成 `已完成` 或 `已结束，但没有全部完成` 时面板自动退出，后者的原因写在日志的最后一行。

取结果（筛出步骤标记、`结果：` 行、HF 贪心表、变好变差、自检和作废的说明，以及和 D-skill 组有没有做成有关的三项：`可用的技能只有` 带出 `可用的技能只有 N 条，少于 MIN_SKILL_TASKS=5…` 这一行；`D-skill 组` 带出 `不做技能引导采样，也不训 D-skill 组` 和最后一行里的 `…没有 D-skill 组`；`没全部写成` 带出 `技能提示没全部写成，继续（没写成的题不做技能引导）`。没读到密钥时第 1 步里那句提醒的头一行（`提醒：环境里没有 TEACHER_API_KEY…`）不在这条命令的输出里，只有它后面两行缩进的说明（都带"D-skill 组"）会被带出来；头一行要用上面那条单独的 grep 看。每行只留前 240 个字符。截断用 Python 而不用 `cut -c`：`cut` 按字节截，中文一行留不下 240 个字。服务器上没有 `python3` 命令时把它换成 `python`）：

<!-- deliver:guided-results -->
```bash
grep -aE '^===== |结果：|一次都没通过|^(模型|base|epoch [0-9])|变好|自检|作废|可用的技能只有|D-skill 组|没全部写成' ~/jupyterlab/CodeAgent-R-v24/rft-guided.log | python3 -c 'import sys; print("\n".join(l[:240] for l in sys.stdin.buffer.read().decode("utf-8", "replace").splitlines()))'
```

贴完应该看到几十行，顺序和日志一样：

- **步骤标记。** 完整的一轮是 `1/13` 到 `13/13`（第 13 步里还有 vLLM 评测自己的 8 步）。有 `可用的技能只有 N 条，少于 MIN_SKILL_TASKS=5` 和 `不做技能引导采样，也不训 D-skill 组` 两行、从 `5/11` 起编号，这一轮就没有 D-skill 组。这几行本身分不清原因：技能写得不行（可用的不到 5 条、一条也没出错）、有题出错（另有 `技能提示没全部写成，继续`）、没读到密钥或写技能整个没做成，都会这样打印。是哪一种看最后一行，见下面的表。
- **引导采样。** 每种提示一行 `结果：… 引导（前缀 0.25）解出 a/72 题（通过样本 p/s）`，前缀的还带照抄了开头的份数；`一次都没通过的题：N 道` 是这种提示引导之后仍然一道都没对的题数。
- **训练和选型。** 每组一张两行的小表和 `结果：<组名> dev 0.xxx → 0.xxx（…；变好 a 道、变差 b 道）`。
- **"汇总主结果"的 HF 贪心表。** `模型` 开头是表头，每组一行 `base`、一行 `epoch 1`；表下每个选中的模型一行 `… 相对 …：变好 a 道、变差 b 道`，后面各组还各有一行恒为 0 的 `base · <组> 相对 …`，不用看。
- **vLLM 两张表。** 只带出 `base` 的行、带"自检"的探针行和两张表下面的结论行，各组的数都在结论行里。

最后一行是 `===== 全部结束 =====`、它前面还有一行 `===== 全部结束 =====`（里面那层 vLLM 评测一行、这一轮一行），说明脚本把它要做的都做完了，但不一定是三组都训了。有没有 D-skill 组、最后一行是哪一种，由"教师给零通过的题写技能提示"那一步的结果决定（表的前提是其余步骤都正常做完）：

| 写技能那一步的情形 | D-skill 组 | 步数 | 最后一行 |
| --- | --- | --- | --- |
| `SKILL=0`：没有这一步 | 没有 | 10（从头就是 `n/10`） | `全部结束` |
| 可用的技能够 5 道（`MIN_SKILL_TASKS`）。个别题出错（有 `技能提示没全部写成，继续` 这一行）也照做，只用写成的题 | 有 | 13 | `全部结束` |
| 可用的不到 5 道，一条也没出错（写技能正常结束） | 没有 | `1/13` 到 `4/13`，接着 `5/11` 到 `11/11` | `全部结束`。这不算故障：技能提示在这批题上不可用，是实验结果，D-prefix 对 F-control 的主结果照常有效 |
| 可用的不到 5 道，又有出错的题 | 没有 | 同上 | `没有全部完成：技能引导没有做成：可用的技能只有 N 条、少于 MIN_SKILL_TASKS=5，其中 M 道出错，没有 D-skill 组；…` |
| 写技能整个没做成（没读到密钥、没装 openai SDK：`write_skills.py` 以退出码 2 结束），可用的不到 5 道 | 没有 | 同上 | `没有全部完成：写技能没有做成：…，没有 D-skill 组；…` |
| 写技能整个没做成，但上一跑留下的可用技能够 5 道 | 有 | 13 | `全部结束` |

任何一行再加上"基座或某一组的 vLLM 评测没评完"，最后一行都是 `===== 没有全部完成：… =====`，原因写在那一行里（可能有两句）；`VLLM_EVAL=0` 时没有这一种，步数各少一步。`没有全部完成`（退出码 1）不影响 dev 上的选型结果，读那一行里的原因，重新运行同一条命令只补没做完的。完整的两张 vLLM 表和"怎么读"在日志的末尾：

```bash
sed -n '/── vLLM 评测/,$p' ~/jupyterlab/CodeAgent-R-v24/rft-guided.log
```

每一步的产物都在 `data/rft-guided/`：

| 文件 | 内容 |
| --- | --- |
| `skills.jsonl` | 教师写的技能，一行一道题：`status` 是 `ok`（可用）、`leaky`（重写到上限仍没通过泄露检查，弃用）或 `error`（请求或解析出错，原样重跑会重做）；另有每一轮的 `checks` 和 `usage`，可用的带正文 `text` |
| `probe-ids.txt` | 族内探针的题号：上一轮零通过的 train 题里属于难题、变体两层的，一行一个 |
| `sampling/train-guided-prefix-*`、`sampling/train-guided-skill-*` | 两种引导采样，文件同第 1 轮（`raw` / `graded` / `candidates` / `protocol` / `manifest`）；manifest 的 `hint` 一节按提示列出槽数、通过样本和解出的题，前缀提示还有照抄了开头的份数 |
| `sft/D-prefix`、`sft/D-skill`、`sft/F-control` | 三组的训练数据；`F-control/unmatched-tasks.txt` 是配不上、各 D 组都不留的题 |
| `checkpoints/<组>`、`results/<组>` | 训练产物和 dev 选型结果，同第 1 轮 |
| `vllm/<标签>/` | vLLM 评测的逐条记录、贪心和采样报告、`summary.json`、自检结果；标签是 `base` 或组名 |
| `vllm-sampling.log`、`vllm-eval.log` | 两次启动 vLLM 的日志 |

常用变量（都有默认值；`MODEL`、`VLLM_BIN`、`RFT_CONCURRENCY` 等其余变量同 `run_rft.sh`）：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `SKILL` | `1` | `1`：教师写技能并做技能引导；`0`：不做，也没有 D-skill 组 |
| `SKILL_JUDGE` | `1` | `1`：写完技能让教师自判一次；`0`：不自判，教师请求少一半，技能只过静态检查。和 `SKILL`、`VLLM_EVAL` 一样只认 `0` 和 `1` |
| `VLLM_EVAL` | `1` | `1`：经本机 vLLM 服务评基座和各组（辅助指标）；`0`：不评 |
| `RFT_TIERS` | `hard,variant` | 引导哪些层的题 |
| `HINT_FRACTIONS` | `0.25,0.5,0.75` | 前缀提示各给参考解的多大比例 |
| `RFT_SAMPLES` | `4` | 前缀引导：每种比例各采几次 |
| `SKILL_SAMPLES` | `8` | 技能引导：每题采几次 |
| `MIN_SKILL_TASKS` | `5` | 可用的技能少于这么多道时，不做技能引导，也没有 D-skill 组 |
| `TEACHER_MAX_ROUNDS`、`TEACHER_MAX_TOKENS`、`TEACHER_TEMPERATURE`、`TEACHER_TIMEOUT` | `2`、`4096`、`0.3`、`120` | 教师接口：每题最多写几轮、每次回复的输出上限、温度、单次请求的超时（秒）。上限 900 时在服务器上成片报"回复里没有 JSON 对象"（多半是没写完就到了上限），所以是 4096；还被截断就继续调大 |
| `TEACHER_CONCURRENCY` | `2` | 写技能时同时写几道题，也就是同一时刻最多有几个教师请求在飞。限额按账户算，服务器上并发 4 时成片 429，所以默认 2；调大以前先想清楚 |
| `TEACHER_MIN_INTERVAL` | `1` | 两次教师请求之间至少隔几秒 |
| `EVAL_SAMPLES` | `4` | vLLM 评测时 dev 上每题采样几次；`0` 表示只做贪心 |
| `BASE_REPORT` | 第 1 轮 A-verified 评出的基座 dev 报告 | 已有的基座 dev 报告。给一个路径就用它；设成空（`BASE_REPORT=`）就由第一组自己评 |
| `RFT_LR`、`RFT_LORA_R`、`RFT_LORA_ALPHA`、`RFT_EPOCHS` | `1e-4`、`32`、`64`、`1` | 三组共用的训练超参 |
| `RUN_DIR`、`ROUND1_DIR`、`TASKS_FILE` | `data/rft-guided`、`data/rft-round1`、`data/frozen-ladder.jsonl` | 这一轮的运行目录、第 1 轮的运行目录、冻结的划分 |

`SKILL=1` 时教师的密钥、地址和模型名从环境变量 `TEACHER_API_KEY`、`TEACHER_BASE_URL`、`TEACHER_MODEL` 读。当前目录有 `.env` 时脚本先载入它，里面的变量会盖过外面同名的环境变量；上表的 `TEACHER_*` 都可以写在里面。可选的 `TEACHER_EXTRA_JSON` 是一个 JSON 对象，原样合并进每次请求（接口特有的参数不用改代码就能传）。没有密钥时（上一跑也没有留下够用的技能）照常做前缀引导，只训 D-prefix 和 F-control 两组，结束时不算是全部完成；把密钥配上再原样重跑，会把它补上。

教师接口会限速：日志里成片的 `教师接口限速（429），等 N 秒后重试（第 k 次）`（服务端出错时是 `教师接口出错（503），等 N 秒后重试（第 k 次）` 这样）是正常的等待，不是出错。脚本自己长时间重试：第一次等 15 秒，之后每次翻倍、最长 120 秒，响应头里有 Retry-After 就按它等，从第一次失败起最多等 30 分钟，到上限才把那一题记成出错。两个脚本同时跑时教师接口会更慢，因为限额按账户算（写技能和合成出题各是一个进程，各自的节流互不知道）。

每一步的产物在就跳过，中断后重新粘贴启动命令就接着跑：已经做完的引导采样、训练数据的准备、各组的训练和选型都不再做（日志里写"已完成，跳过"或"已经选过型，跳过"），vLLM 也只在还有活要干时才启动；写技能在技能引导采样开始以前每次都会调用，它只补没写完的题。采样或评测途中 vLLM 退出（显存不足、被系统杀掉）时，脚本把它重新启动一次再接着做，每一步最多重启一次。最后一行是 `===== 没有全部完成：… =====`（退出码 1）时，脚本已经走到最后，只是有没做完的：技能引导没做成（没有密钥或写技能整个失败；或者可用的技能不到 5 条、而且其中有出错的题），或者基座、某一组的 vLLM 评测没评完。dev 上的选型结果不受影响，重新运行同一条命令只补没做完的。可用的技能不到 5 条、一条也没出错、写技能也正常结束时不属于这一种：脚本同样不做 D-skill 组，却以"全部结束"收尾（见"取结果"里的表）。教师写技能的回复被截断时，报错里会提示把 `TEACHER_MAX_TOKENS` 调大（写在 `.env` 里），调大后原样重跑只重做出错的题；前提是技能引导采样还没有开始过（见下面的两条规矩）。

脚本在半路停下时（退出码 1，没走到最后），原因和做法都印在日志里：

| 日志里的原因 | 怎么办 |
| --- | --- |
| `显卡被占用` | 上一轮的 vLLM 或评测还占着显卡（占着它的进程会列出来）。停掉它再重新粘贴启动命令 |
| `vLLM 没起来`，后面是 vLLM 日志的最后 15 行 | 显存不够或环境不对。完整的日志在 `data/rft-guided/vllm-sampling.log` 或 `vllm-eval.log` |
| `缺少任务文件` | 上一包的 `data` 没接过来（上一包不叫 `CodeAgent-R-v19` 时，改启动命令里 `ln -s ../CodeAgent-R-v19/data data` 这一处），或者 `TASKS_FILE` 指的不对 |
| `缺少第 1 轮的产物` | `data` 接上了，但第 1 轮没做完；缺的是哪几个文件会列出来 |
| `找不到 vLLM` | vLLM 不在默认路径 `~/vllm-qwen/bin/vllm`，用 `VLLM_BIN` 指定 |
| `没有可引导的题` | 上一轮零通过的题里没有 `RFT_TIERS` 写的那几层 |
| `sample_rft.py 以退出码 2 结束` | 采样参数和这个目录里已有的协议对不上（两次运行之间改了 `RFT_SAMPLES`、`SKILL_SAMPLES` 之类）。把改动过的变量改回去，或者换一个新的 `RUN_DIR` |
| `配对对照没做成` | 引导采样没有解出第 1 轮验证组之外的新题，这一轮没有可比较的引导组 |
| `三次都没完成` | 采样连续三次没跑完，后面是 vLLM 日志的最后 15 行 |
| `训练数据没通过检查`、`训练出错`、`选型出错` | 后面是 Python 的报错原文 |

表以外的停法，日志里同样写了原因；后面跟着 Python 报错原文的，把报错和日志的最后几十行一起留着。

两条和重跑有关的规矩。技能引导采样一旦开始过，`skills.jsonl` 就不再改动（采样协议里记着它的摘要），没写成的题这一轮不做技能引导。两次运行之间 `SKILL` 变了（上一跑没有密钥，或者这一跑设了 `SKILL=0`），F-control 按"现在有几个引导组"重新配对、重新训练，原来的数据、检查点、选型结果和 vLLM 评测挪到名字后面加 `.unpaired-时间戳` 的目录。

## vLLM 评测（辅助指标）

主协议是 `scripts/evaluate.py`：transformers 加载 NF4 量化的模型，每道题贪心生成一次，dev 选型和写结论都用它。它慢（66 道 dev 题约半小时），而且只看一次贪心，对"训练有没有起作用"不够灵敏：一道题采样几次能对几次，贪心那一次没对就仍然算错。所以引导采样这一轮在主结果之外，用 `scripts/eval_vllm.py` 经本机 vLLM 服务把基座和各组选中的适配器再评一遍（dev 上贪心加每题采样 4 次，探针上只做贪心），`scripts/vllm_summary.py` 把结果排成两张表。它只作旁证，`VLLM_EVAL=0` 可以整段关掉；`run_rft.sh` 的 `VLLM_EVAL=1` 用的是同一套脚本，只评 dev，第 1 轮没有探针，也不自检。

| | 主结果 | 辅助指标 |
| --- | --- | --- |
| 脚本 | `scripts/evaluate.py`（`select_checkpoint.py` 调它） | `scripts/eval_vllm.py` |
| 权重 | NF4 量化 | bf16，不量化 |
| 每题生成 | 贪心 1 次 | 贪心 1 次，再采样 4 次（temperature 0.7、top_p 0.95、top_k 20，上限 2048 token） |
| 指标 | `pass@1` | 贪心的 `pass@1`；采样按次数算的通过率，以及至少通过一次的题占多少（pass@4） |
| 报告里的记号 | `generation.backend` 是 `transformers`，`generation.quantization` 是 `nf4` | `generation.backend` 是 `vllm`，`generation.quantization` 是 `bf16` |

两边的权重精度、提示的渲染方式都不同，同一个模型在两边得到的通过数可以不一样，数字不能混用：vLLM 的数只和 vLLM 的数比（各组对基座），主结果那张表里不出现它。

**自检。** 适配器是 vLLM 启动时用 `--lora-modules 组名=路径` 挂上去的。它没挂上时，评出来的其实是基座，数字看上去也没有异常。所以 D 组在评探针之前先自检：从这次要评的题里取几道这一组训练过的题（最多 6 道），适配器和同一条服务上的基座各贪心生成一次，逐题比较两边的输出。输出和基座不同的题一道都没有（门槛 `--sanity-min`，默认 1）就不往后评，这一组的 vLLM 评测作废，汇总表里写"评测作废或没做"和原因；这是作废的唯一原因。自检同时统计训练过的题严格通过了几道，只作参考：只训 1 个 epoch、约 100 条数据时，贪心解码复现不了训练答案是正常的，和适配器有没有生效无关。拿不到基座名（服务的模型列表里分不出适配器挂在哪个基座上）、或者要评的题里一道训练过的都没有时没法比较：不作废，照常往下评，自检结果（`*-sanity.json` 里的 `compared` 和 `not_compared_reason`）和汇总表里都注明"没能确认适配器是否生效"。F-control 是在没通过的答案上训练的，不自检。被作废的组不影响这一轮算不算做完，它在 HF 贪心表里的那一行照常有效。第一次在服务器上评适配器时，看一眼日志里每次 `eval_vllm.py` 开头的那一行：应该是 `服务 … 上的 D-prefix 是挂在基座 … 上的适配器`；写成"是基座"就说明这版 vLLM 没有报告适配器的归属，报告里的模型名不可信；写成"是适配器（模型列表里分不出它挂在哪个基座上）"时自检没法比较，这一组只能算没能确认。

**族内探针。** 探针就是上一轮零通过的 72 道题（`data/rft-guided/probe-ids.txt`），基座和每组的适配器在它们上各贪心一次。汇总时按这一组训练过没有分成两项，括号里是基座在同一批题上的通过数：

- 训练过的题通过得很少：模型连训练时见过的正确答案都没学会，训练信号太弱（先看 D 组训练集里引导题有几条）。
- 训练过的题过了不少，没训练过的同族题和基座一样：这几道题学进去了，但跨不了族。

dev 那张表（表一）每行是一个标签的贪心、采样通过率和 pass@4，下一行是各层的贪心结果。采样通过率比贪心灵敏，但只作旁证，下结论仍以 HF 贪心表为准。dev 上保留了基座的组没有适配器，不用评。只想重看这两张表时不必开 vLLM，在 `~/jupyterlab/CodeAgent-R-v24` 里运行：

```bash
python scripts/vllm_summary.py --run data/rft-guided --arms D-prefix D-skill F-control --probe-ids-file data/rft-guided/probe-ids.txt
```

## 合成出题：用教师接口扩题

题太少是根本瓶颈：题池只有 327 道题，难题只有 20 个来源族，第 1 轮 RFT 的训练数据只有 61 条，看不出提升。`scripts/run_synth_pool.sh` 让教师接口成批地出新题，每一道都在沙箱里真判过才收，一批一批并进阶梯文件。它只用 CPU 和教师接口，不碰显卡，可以和显卡上的实验同时跑（两边都在判分，CPU 会分摊）。

**题族是什么。** 一个题族就是一个程序、一个来源族，结构照现有的难题：一道母题 `s3_<主题>_001`，程序里同时带着多处缺陷；每个可用的缺陷另有一道单缺陷变体 `s3v_<主题>_<k>`，题面、参考解、测试都同母题，程序里只留这一处。母题属于"合成难题"层（`synth_hard`），变体属于"合成变体"层（`synth_variant`），在采样汇总和报告里各占一栏，排在原有三层后面。主题是"出题方向 × 具体领域"：五个方向（状态一致性、事务回滚、异常安全、锁序、条件变量）各配 109 个领域（库存、账本、缓存……），一共 545 个；一批 50 个主题，每个方向 10 个，够出 10 批。主题的短名（例如 `state_inventory`）就是记录的文件名，报错里把它叫 slug。

**流程。** 每个主题向教师要一个题族，最多两轮：

1. 出题。提示里有这个方向的说明和这类题常见的疏忽、领域、这道题要有几条编号的要求（4 到 6 条，由主题决定）、一道同方向现有难题的风格参照，以及要交回来的东西：完全正确的参考解、题面、公开测试、按标签组织的隐藏测试、若干 mutant（每个把参考解的一处改坏，对应一条要求）。
2. 审计。回复解析以后在沙箱里真跑，八项都查完才下结论。
3. 返修。没被接受、但问题可以返修时，把上一版和查出来的问题逐条给回去，让它改一轮；还不行就拒绝。回复被截断、不成 JSON 也算一轮，下一轮带着"上次被截断"（或者"上次不是合法的 JSON"）请它重写。撞车这类返修解决不了的问题不再花一轮；教师调用出错时下一轮重新出题。

接受的题族 = 母题 1 道 + 每个可用的 mutant 一道变体。可用的 mutant 不到 3 个时母题不成立，整个题族拒绝。

**审计查什么。** 第 2 到 5 项在沙箱里真跑 pytest，"被拦住"只认隐藏测试里的断言失败，全部通过、报错、超时、进程异常退出都不算；其余几项是不跑代码的静态检查：

1. 结构。题目的规格合法；题号、来源组、模块名不和内置题池及这一池里已接受的题族相同，原程序不和它们的语法树相同，参考解不和这一池里已接受题族的参考解语法树相同（撞了只能换主题，返修解决不了）；每个 mutant 的 `old` 在参考解里恰好出现一次，改坏以后的程序仍是合法的 Python，不和参考解或别的 mutant 改坏以后的程序相同，mutant 之间改动不重叠；测试文件里有 pytest 收得到的测试函数。
2. 参考解。公开测试和隐藏测试全部通过，重复 2 次结论一致。
3. 每个单缺陷变体。被隐藏测试的断言拦住，公开测试仍然通过。
4. 母题（带着全部可用的缺陷）。同样被隐藏测试的断言拦住、公开测试通过，重复 2 次结论一致。
5. 隐藏测试能区分各条要求：不同变体挂掉的测试不全一样。
6. 长度。母题的提示加参考解粗估不超过 1900 个 token（训练默认长度是 2048，超长的样本会被整条丢掉），参考解 30 到 70 行。
7. 安全。没有起子进程、联网、读写文件的模块和调用，没有 `eval`、`exec`、`compile`、`__import__`、`input`；`time.sleep` 的参数只能是不超过 0.2 秒的字面量。
8. 没有提示缺陷在哪的注释标记（`BUG`、`FIXME`、`TODO`、`XXX`、`HACK`）。

被接受要同时满足：没有撞车这类返修解决不了的问题，题目的规格合法；第 2、4、6、7 项过关；可用的 mutant 不少于 3 个。不合格的 mutant（结构里查出的，或者第 3 项里没被拦住的）直接剔除，不卡接受：剩下的够 3 个就照收，只用可用的出变体。第 5、8 项只把问题记下来，也不影响接受，所以收进来的题族仍可能有区分度不够的测试或者带标记的注释（记在它那份记录的 `findings` 里）。第 7 项只看语法树上写得出来的写法（`import`、直接的调用）；这台服务器的内核不支持 Landlock，脚本也没有要求必须有沙箱，教师写的程序和测试是在退化后的隔离（只有 rlimit 和清洁环境，文件系统与网络不受限）里执行的。

**划分和阶梯。** 每批的最后一步把到现在为止接受的全部题族冻结划分并导出。划分以来源族为单位：只给还没冻结过的题族分 train / dev / test，比例 20% / 20%（对族数向下取整，按五个方向分层），冻结过的不再变；母题和变体属于同一个来源族，永远在同一个划分里。一次新接受的题族不到 3 个时分不出三个划分，这几族都记在 train 里，以后也不再改。导出的新题文件是全部合成题；阶梯文件 `data/frozen-ladder-v2.jsonl` 每次都改写成"原来的 `data/frozen-ladder.jsonl`（原有的题一行不动）+ 到现在为止的全部合成题"。**出完所有批次以前不要开始基于新阶梯的训练**：每出一批这个文件都会变，训练做到一半时题变了，各步读到的就不是同一份题。

**花多少。** 一批 50 族是 50 到 100 次教师调用：每族第一轮 1 次，没被接受、问题又可以返修的再加 1 次；一半要返修就是 75 次左右。时间按实现推：同时最多 2 个请求在飞、两次请求至少隔 1 秒；教师的回复带推理、有好几千个 token，一次按一到几分钟算；审计每个题族要在沙箱里判分十来次（参考解 2 遍、每个变体 1 遍、母题 2 遍），审计之间串行，一批一般一共十几二十分钟，有变体让测试超时（每个最多 30 秒）就更久。合起来一批大约几个小时，教师慢或者成片限速时更久。这是估计，还没有在服务器上跑过；第一批跑完，看结果行里的"教师调用 N 次、用时 T 分钟"就准了。

教师接口会限速：日志里出现 `教师接口限速（429），等 N 秒后重试（第 k 次）` 是正常的，不是出错，脚本自己会等到恢复（最多 30 分钟；间隔见"引导采样的一轮"里环境变量下面那一段）。

### 启动、看进度、取结果

教师的密钥从 `CodeAgent-R-v24` 目录里的 `.env` 读（`TEACHER_API_KEY`、`TEACHER_BASE_URL`、`TEACHER_MODEL`）。引导那一轮的启动命令已经把它拷过去了，所以先贴那一条；还没贴过就先把 `.env` 放进 `~/jupyterlab/CodeAgent-R-v24/`。先跑这一条也可以：`data` 还没接过来时脚本会自己把它接到上一包的 `data`（和引导那一轮共用）；`data` 已经是本包里的真实目录、里面又没有阶梯文件时，脚本会停下并说明怎么处理。下面这一条命令在后台启动第 1 批；重复粘贴不会起第二个：

<!-- deliver:synth-launch -->
```bash
cd ~/jupyterlab/CodeAgent-R-v24 && if pgrep -f 'run_synth_pool[.]sh' > /dev/null; then echo "已经在运行，不重复启动"; else nohup bash scripts/run_synth_pool.sh > synth-batch1.log 2>&1 < /dev/null & echo "已启动"; fi
```

贴完应该看到 `已启动`（前面可能多一行 `[1] 进程号`；跑完以后回到终端，可能还多一行 `[1]+ Done …`）。脚本已经在跑时看到的是 `已经在运行，不重复启动`，正在跑的那份不受影响；这一句只说明有哪个进程的命令行里带着脚本名，另一个终端里的进度面板还没退出时也会这样，等它退出再粘。日志是 `~/jupyterlab/CodeAgent-R-v24/synth-batch1.log`：每次启动都会重写它，上一次的想留着就先拷走（`data/synth-pool/batch-N.log` 也只留最近一次向教师要题的输出）。没读到密钥时它只有三行：步骤标记 `===== 1/5 检查前提 =====` 和 `环境里没有 TEACHER_API_KEY：把 .env 放到项目目录（仓库根，和 scripts 并列），里面写 TEACHER_API_KEY=…；` 的两行说明，什么都没请求：把 `.env` 放进这个目录再粘一次。读到密钥时开头三行是：

```
===== 1/5 检查前提 =====
第 1 批：向教师要 50 个题族（方向 all），记录在 data/synth-pool
===== 2/5 向教师要题族并逐个审计（每族最多 2 轮）=====
```

接着是一行 `第 1 批：向教师要 50 个题族（全部方向）；每个最多 2 轮，同时 2 个、请求至少隔 1 秒；目录里已有定论的题族 0 个`，然后每做完一个题族多一行进度，例如 `[#########---------------] 20/50 (40%) cond_coupon: 接受（第 2 轮，变体 4 个） · 已用 23m`，拒绝的写成 `拒绝（原因）`。日志里还会有一行 `警告：Landlock 不可用（…）`，是上面说过的沙箱退化，只出现一次。

看进度（每 60 秒刷新一次，这一批结束后自动回到提示符；Ctrl+C 只是停止观看，不影响后台任务）：

<!-- deliver:synth-progress -->
```bash
cd ~/jupyterlab/CodeAgent-R-v24 && WATCH=60 bash scripts/show_progress.sh synth-batch1.log '结果：|接受|拒绝' run_synth_pool.sh
```

贴完应该看到一块面板：`状态: 进行中`；`步骤` 是五步（`1/5 检查前提` 到 `5/5 并入阶梯`），做完的标着 `[完成]`，跑完以后最后一行 `全部结束` 也列在这里；`当前进度` 是最后几行进度条；`已出的结果` 是到这时为止带"接受""拒绝"或者以 `结果：` 开头的行（只留最后 14 行），也就是每个题族的结论和后面的结果行；最下面是 `日志最后 3 行`。状态变成 `已完成` 时面板自动退出，说明五步都走完了；变成 `没有在运行（中途中断了，看最后几行找原因）` 是第 2 步没做完就停了，原因在日志的最后几行。

取结果（筛出步骤标记、以 `结果：` 开头的行、带"接受"或"拒绝"的行，每行只留前 240 个字符。截断用 Python 而不用 `cut -c`：`cut` 按字节截，中文一行留不下 240 个字。服务器上没有 `python3` 命令时把它换成 `python`）：

<!-- deliver:synth-results -->
```bash
grep -aE '^===== |结果：|接受|拒绝' ~/jupyterlab/CodeAgent-R-v24/synth-batch1.log | python3 -c 'import sys; print("\n".join(l[:240] for l in sys.stdin.buffer.read().decode("utf-8", "replace").splitlines()))'
```

贴完应该看到六七十行，顺序和日志一样：

- **步骤标记。** `1/5` 到 `5/5`，最后是 `===== 全部结束 =====`。
- **每个题族一行进度。** `[#########---------------] 20/50 (40%) cond_coupon: 接受（第 2 轮，变体 4 个） · 已用 23m`，拒绝的是 `…: 拒绝（原因）`。
- **这一批的结果行。** `结果：第 1 批 接受 N 族（母题 N 道、变体 M 道）、拒绝 K 族（原因：…）、教师调用 C 次、用时 T 分钟`，原因按数量从多到少列。
- **汇总。** `===== 3/5 汇总这一池 =====` 后是 `结果：共接受 … 族（母题 …、变体 …）、拒绝 … 族；各方向 状态一致性 a/b、…`（到现在为止所有批次合在一起）和 `拒绝原因：…`。
- **抽查。** `===== 4/5 抽查 3 个已接受的题族 =====` 后是抽到的题族的标题行 `── 1/3 <主题>（状态一致性 · 库存；第 1 轮接受，母题 1 道、变体 4 道）──` 和 `结果：抽查 3 个已接受的题族（一共 … 个；…）`。题面和 diff 里带"接受""拒绝"字样的行也会被带出来，是正常的。
- **导出。** `===== 5/5 并入阶梯 =====` 后是 `结果：导出 N 道合成题（母题 a 道、变体 b 道；这次新冻结 f 族）：train … 道、dev … 道、test … 道；读回来核对过，没有来源组跨划分`。

`5/5` 后面没有 `结果：导出 …` 这一行，就是导出没有完成（脚本照样以"全部结束"收尾）。下一批怎么跑、有没有没做完的题族，写在第 2 步的末尾，不在这条命令的输出里。这两样和出错的说明用下面这一条看：

```bash
grep -n -E '下一批这样跑|没有定论|导出|并入阶梯|没有跑完|没有一族能用|一个都没接受|全部出错|没打印出来|这一批停下了|写不进|立刻停下|TEACHER_API_KEY|排不出' ~/jupyterlab/CodeAgent-R-v24/synth-batch1.log
```

一批正常跑完时带出来的是：`===== 5/5 并入阶梯 =====`；第 2 步末尾的 `下一批这样跑：BATCH=2 bash scripts/run_synth_pool.sh`；`结果：导出 …`；`并入阶梯：data/frozen-ladder-v2.jsonl = data/frozen-ladder.jsonl（原有的 327 道原样保留）+ 新题 N 道`。有没定论的题族时（教师调用出错、回复被截断或者没法解析、判分没能完成，见下面"续跑"）还多两行：第 2 步末尾的 `其中 N 族没有定论，不算用掉。只补它们、不要新主题：FAMILIES=N BATCH=1 bash scripts/run_synth_pool.sh`（N 是这一批里的），和汇总那一步的 `其中 M 族没有定论（…）…`（M 是整个题池里的）。别的行都是出了事，对下面的表。

### 常用变量

都有默认值。写在命令里 `nohup` 的前面（像下面"出下一批"那样），或者写进 `.env`；教师的几个参数适合写在 `.env` 里，一批一变的 `BATCH`、`FAMILIES` 不要写进去。

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `BATCH` | `1` | 第几批，从 1 起 |
| `FAMILIES` | `50` | 这一批向教师要多少个题族。被拒绝的不补：它是"要多少个"，不是"接受到多少个" |
| `DIRECTIONS` | `all` | 出题方向：`all`，或者逗号分隔的方向短名（`state`、`tx`、`exc`、`lock`、`cond`） |
| `SYNTH_DIR` | `data/synth-pool` | 题族的记录、冻结的划分、新题文件放在这里 |
| `LADDER_FILE` | `data/frozen-ladder.jsonl` | 已有的阶梯文件。它不存在（或是空文件）时跳过导出，日志里会说明，题族的记录照常保留 |
| `LADDER_OUT` | `data/frozen-ladder-v2.jsonl` | 阶梯加上新题写到哪。里面要是别的内容（不是这个脚本以前导出的）就不覆盖 |
| `TEACHER_CONCURRENCY` | `2` | 同时处理几个题族，也就是同一时刻最多有几个教师请求在飞。限额按账户算，调大以前先想清楚 |
| `TEACHER_MIN_INTERVAL` | `1` | 两次教师请求之间至少隔几秒 |
| `TEACHER_MAX_TOKENS` | `12000` | 每次回复的输出上限。一个题族的 JSON 本身就有一两千个 token，回复被截断时调大它 |
| `TEACHER_TEMPERATURE`、`TEACHER_TIMEOUT` | `0.8`、`300` | 教师接口的温度、单次请求的超时（秒） |

`TEACHER_API_KEY`、`TEACHER_BASE_URL`、`TEACHER_MODEL`、`TEACHER_EXTRA_JSON` 的读法同"引导采样的一轮"。

### 产物

都在 `data/synth-pool/` 下（`data` 接的是上一包的 `data`，所以实际在它的 `data` 目录里）：

| 文件 | 内容 |
| --- | --- |
| `families/<主题>.json` | 每个主题一份记录：`status`（`accepted` 或 `rejected`）、拒绝时的 `reason`、`rounds`（每一轮是出题还是返修、教师的原始回复、结束原因、用量、审计的发现和统计），接受时还有 `family`（母题和变体的全部题目） |
| `batch-N.log` | 第 N 批向教师要题、审计时的输出，同 `synth-batchN.log` 的第 2 步 |
| `batches.done` | 正常跑完的批次，一行一批。原样重跑其中某一批时，第 2 步直接跳过 |
| `splits.json` | 每个来源族冻结在哪个划分，以及每次冻结的记录 |
| `synth-pool.jsonl` | 到现在为止接受的全部合成题，一行一道，带 `split` |
| `synth-pool.audit.json` | 每个题族请求了几轮、用了多少 token，每道题的审计结论（变体对应哪条要求、被哪些测试拦住） |

### 出下一批

一批跑完（面板显示 `已完成`）再启动下一批：两条启动命令同时贴没有意义，前一条的 `pgrep` 和脚本里的锁都会拦住后一个。把启动命令改两处：日志名改成 `synth-batch2.log`，并在 `nohup` 的前面加上 `BATCH=2 `：

```bash
cd ~/jupyterlab/CodeAgent-R-v24 && if pgrep -f 'run_synth_pool[.]sh' > /dev/null; then echo "已经在运行，不重复启动"; else BATCH=2 nohup bash scripts/run_synth_pool.sh > synth-batch2.log 2>&1 < /dev/null & echo "已启动"; fi
```

`BATCH=2` 必须紧挨着写在 `nohup` 前面。写成 `nohup BATCH=2 bash …` 时 nohup 会把 `BATCH=2` 当成要运行的程序，报 `failed to run command`；写在这一整条的最前面也不行，那样它只作用到 `cd` 上。看进度、取结果的两条命令把 `synth-batch1.log` 换成 `synth-batch2.log`，其余不动。第 3 批以后照此类推。主题一共 545 个，第 10 批以后还剩 45 个，最后一批写 `BATCH=11 FAMILIES=45`（照 50 个要会报 `排不出第 11 批的主题`）。

### 中途停了怎么办、怎么续跑

续跑就是原样再粘一次同一条启动命令，什么都不用改：

- 某一批用同样的参数正常跑完过，第 2 步直接跳过（`这一批用同样的参数正常跑完过，跳过；再出一批用 BATCH=N+1。`），只重做后面的汇总、抽查和导出，不会再向教师要题，多粘一次不多花钱。
- 中途被杀或者出错停下，没做完的题族排在最前，再往后要新的，把这一批凑满。已经有定论的题族（接受了，或者因为题族本身被拒绝）不再请求。教师调用出错、回复被截断或者没法解析、判分没能完成的题族不算定论，重跑时会再要一次，新记录盖掉旧的。
- 一批正常跑完、但有几个题族没定论时，日志里会有 `其中 N 族没有定论，不算用掉。只补它们、不要新主题：FAMILIES=N BATCH=1 bash scripts/run_synth_pool.sh`。想补就照它写：把 `FAMILIES=N` 和 `BATCH=1` 一起写在 `nohup` 前面再贴一次，只补这 N 族；不补、直接出下一批也行，这几个主题就空着了。

要停掉正在跑的一批：

```bash
pkill -f run_synth_pool.sh
```

立刻停，正在做的那几个题族的回答不保存（日志里是 `收到 kill 信号：立刻停下。…`），重跑会再要。正在看的进度面板也会被它关掉（它的命令行里也带着脚本名），不用管。

出错了照下面的表处理，原因都写在日志里：

| 日志里的行 | 意思 | 怎么办 |
| --- | --- | --- |
| `第 N 批没有跑完（退出码 3）`，上面是 `教师有回复，但没有一族能用（…）` | 一族都没接受，原因是回复被截断或者没法解析 | 照那句话做：被截断就把 `TEACHER_MAX_TOKENS` 调大（写进 `.env`，或者和 `BATCH` 一样写在 `nohup` 前面）；没法解析就换 `TEACHER_MODEL`，或者把 `TEACHER_TEMPERATURE` 调低。然后原样重跑：是把这一批的主题再要一次，不会要新主题 |
| `第 N 批没有跑完（退出码 3）`，上面是 `教师调用全部出错：检查 TEACHER_API_KEY、TEACHER_BASE_URL 和网络` | 教师接口一次都没连上 | 检查 `.env` 里的密钥和地址、网络，修好以后原样重跑 |
| `第 N 批没有跑完（退出码 3）`，上面是 `这一批一个都没接受（判分没能完成 N 族…）`；或者 `连续 3 个题族都因为判分没能完成或者程序出错被拒，这一批停下了：…`（这时退出码是 1） | 判分环境有问题：要么一族都没审完，要么连着出错，脚本不再调用教师 | 先查沙箱和判分环境（`python -m agentlab doctor`），修好以后原样重跑，这些题族会再要一次 |
| `第 N 批没有跑完（退出码 1）`，上面是 `收到 kill 信号：立刻停下。…` | 被 `pkill` 停掉了 | 不用处理，原样重跑就接着做 |
| `有 N 族的记录写不进 …` | 磁盘满了，或者目录只读；这一批停下了 | 腾出空间或者改好权限，原样重跑 |
| `导出没有完成（…）` | 并入阶梯这一步失败。题族的记录都还在，这一批算跑完，脚本照样打印"全部结束" | 看括号里的原因（多半是 `data/frozen-ladder-v2.jsonl` 里是别的内容）。处理以后原样重跑，这一批不会再向教师要题 |
| `排不出第 N 批的主题：…` | 主题用完了 | 剩下的按提示的个数写 `FAMILIES=` |

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
