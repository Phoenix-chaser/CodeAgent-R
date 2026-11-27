#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
[[ ! -f "$HOME/setup_env.sh" ]] || source "$HOME/setup_env.sh"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

STAGE="${1:-all}"
FORCE="${FORCE:-0}"
MODEL="${MODEL:-}"                    # 留空则用 src/agentlab/models.py 里的默认模型
MODEL_REVISION="${MODEL_REVISION:-}"
DEV_RATIO="${DEV_RATIO:-0.15}"
TEST_RATIO="${TEST_RATIO:-0.15}"
TRAIN_CANDIDATES="${TRAIN_CANDIDATES:-12}"
DEV_CANDIDATES="${DEV_CANDIDATES:-4}"
MAX_PER_TASK="${MAX_PER_TASK:-1}"
EVAL_SPLIT="${EVAL_SPLIT:-dev}"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
QUANTIZATION="${QUANTIZATION:-nf4}"
EXTRA_POOLS="${EXTRA_POOLS:-}"      # 可选题池，如 h3（难题）；留空就是默认的 104 题
export MODEL_REVISION TRAIN_CANDIDATES DEV_CANDIDATES MAX_PER_TASK SEED MAX_NEW_TOKENS QUANTIZATION
export EXTRA_POOLS
# 启用了可选题池（里面有难题）时按难度分层划分，让难题的来源组按比例进入三个集合
SPLIT_STRATIFY="${SPLIT_STRATIFY:-$([[ -n "$EXTRA_POOLS" ]] && echo 1 || echo 0)}"
# 思考模式（Qwen3.5 的默认行为）。只影响 select / eval / final-eval：回答在 </think> 之后，
# 推理会占用生成额度，所以要同时调大 MAX_NEW_TOKENS
THINKING="${THINKING:-0}"
EVAL_FLAGS=()
[[ "$THINKING" != 1 ]] || EVAL_FLAGS=(--thinking)
pyrun() {
  if command -v python >/dev/null 2>&1; then python "$@"; else python3 "$@"; fi
}
state() { pyrun src/agentlab/experiments/run_state.py "$@"; }
# 默认模型只在 src/agentlab/models.py 定义一次
MODEL="${MODEL:-$(state default-model)}"
export MODEL
# revision 与模型成对：换了模型却沿用旧 revision，会按旧模型的提交去拉取新模型
if [[ -z "$MODEL_REVISION" ]]; then
  MODEL_REVISION="$(state revision "$MODEL")"
  export MODEL_REVISION
  [[ -n "$MODEL_REVISION" ]] || echo "警告：$MODEL 没有已核对的固定 revision，将使用 main；复现实验请显式设置 MODEL_REVISION。" >&2
fi
paths="$(state init)"
IFS=$'\t' read -r RUN_ID STATE_DIR EXP_DIR CKPT_DIR SFT_DIR EVAL_OUTPUT_DIR TRAIN_CANDIDATES_FILE DEV_CANDIDATES_FILE <<< "$paths"
POOL_DIR="$STATE_DIR/pools"
LOG_DIR="$STATE_DIR/logs"
SELECTION_DIR="$EVAL_OUTPUT_DIR/selection"
RUN_TAG="$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$LOG_DIR"
TRAIN_ARGS=(--epochs "${EPOCHS:-3}" --lr "${LR:-2e-5}" --batch-size "${BATCH_SIZE:-1}"
            --grad-accum "${GRAD_ACCUM:-8}" --max-length "${MAX_LENGTH:-2048}"
            --lora-r "${LORA_R:-16}" --lora-alpha "${LORA_ALPHA:-32}"
            --lora-dropout "${LORA_DROPOUT:-0.05}" --patience "${PATIENCE:-2}"
            --min-delta "${MIN_DELTA:-0.001}" --seed "$SEED")

if [[ -z "${TEACHER_API_KEY:-}" && -f .env ]]; then
  TEACHER_API_KEY="$(pyrun - <<'PY'
from pathlib import Path
for line in Path('.env').read_text().splitlines():
    if line.startswith('TEACHER_API_KEY='):
        print(line.split('=', 1)[1].strip().strip('\"\''))
        break
PY
)"
  export TEACHER_API_KEY
fi

echo "CodeAgent-R run=$RUN_ID stage=$STAGE"
echo "模型: $MODEL @ $MODEL_REVISION"
echo "实验: $EXP_DIR | checkpoint: $CKPT_DIR | 结果: $EVAL_OUTPUT_DIR"

artifacts() {
  INPUTS=("$STATE_DIR/configuration.json" scripts/run_experiment_v2.sh src/agentlab/experiments/run_state.py)
  OUTPUTS=()
  case "$1" in
    select|eval|final-eval)
      # 思考模式不进入 run_id；打开后写进请求文件，参与这几个阶段的回执校验，
      # 关闭时不写，默认流程的回执与改动前完全相同
      [[ "$THINKING" != 1 ]] || printf 'thinking=1\nmax_new_tokens=%s\n' "$MAX_NEW_TOKENS" \
        > "$STATE_DIR/eval-request.txt" ;;
  esac
  case "$1" in
    pools)
      INPUTS+=(scripts/export_pilot_tasks.py scripts/export_tasks_v2.py src/agentlab/benchmark)
      OUTPUTS=("$POOL_DIR/combined.jsonl") ;;
    split)
      # 划分比例不进入 run_id（已有状态目录继续有效），但要参与 split 阶段的回执校验：
      # 改了比例不能静默复用按旧比例冻结的划分
      printf 'seed=%s\ndev_ratio=%s\ntest_ratio=%s\n' "$SEED" "$DEV_RATIO" "$TEST_RATIO" \
        > "$STATE_DIR/split-request.txt"
      # 是否分层同样不进入 run_id，但改变了就不能复用按另一种方式冻结的划分
      [[ "$SPLIT_STRATIFY" != 1 ]] || echo 'stratify=1' >> "$STATE_DIR/split-request.txt"
      INPUTS+=("$STATE_DIR/split-request.txt" "$POOL_DIR/combined.jsonl" scripts/prepare_data.py
               src/agentlab/training/preparation.py)
      OUTPUTS=("$EXP_DIR/tasks.jsonl" "$EXP_DIR/split-manifest.json" "$EXP_DIR/audit-manifest.json") ;;
    collect)
      INPUTS+=("$EXP_DIR/tasks.jsonl" scripts/collect_data.py src/agentlab/benchmark)
      OUTPUTS=("$TRAIN_CANDIDATES_FILE" "${TRAIN_CANDIDATES_FILE%.jsonl}.manifest.json"
               "$DEV_CANDIDATES_FILE" "${DEV_CANDIDATES_FILE%.jsonl}.manifest.json") ;;
    prepare)
      INPUTS+=("$EXP_DIR/tasks.jsonl" "$TRAIN_CANDIDATES_FILE" "$DEV_CANDIDATES_FILE"
               scripts/prepare_data.py scripts/collect_data.py src/agentlab/training/preparation.py
               src/agentlab/benchmark)
      OUTPUTS=("$SFT_DIR/train.jsonl" "$SFT_DIR/dev.jsonl" "$SFT_DIR/dataset-manifest.json") ;;
    train)
      INPUTS+=("$SFT_DIR/train.jsonl" "$SFT_DIR/dev.jsonl" scripts/train_sft.py src/agentlab/training/hf_sft.py)
      OUTPUTS=("$CKPT_DIR/training-manifest.json" "$CKPT_DIR/training-log.jsonl"
               "$CKPT_DIR/adapter" "$CKPT_DIR/checkpoint-index.json") ;;
    select)
      INPUTS+=("$EXP_DIR/tasks.jsonl" "$CKPT_DIR" scripts/select_checkpoint.py scripts/evaluate.py
               scripts/_shared_extract.py scripts/compare.py src/agentlab/benchmark)
      [[ "$THINKING" != 1 ]] || INPUTS+=("$STATE_DIR/eval-request.txt")
      OUTPUTS=("$SELECTION_DIR") ;;
    eval|final-eval)
      INPUTS+=("$EXP_DIR/tasks.jsonl" "$CKPT_DIR" "$SELECTION_DIR"
               scripts/evaluate.py scripts/_shared_extract.py scripts/compare.py src/agentlab/benchmark)
      [[ "$THINKING" != 1 ]] || INPUTS+=("$STATE_DIR/eval-request.txt")
      local split=dev
      [[ "$1" != final-eval ]] || split=test
      OUTPUTS=("$EVAL_OUTPUT_DIR/$split/base.json" "$EVAL_OUTPUT_DIR/$split/selected.json"
               "$EVAL_OUTPUT_DIR/$split/comparison.md") ;;
  esac
}

run_guarded() {
  local name="$1"; shift
  if [[ "$name" != deps && "$name" != env && "$name" != smoke ]]; then
    artifacts "$name"
    if [[ "$FORCE" != 1 ]] && state check --receipt "$STATE_DIR/$name.json" \
      --inputs "${INPUTS[@]}" --outputs "${OUTPUTS[@]}"; then
      echo "── [$name] 内容与配置已验证，跳过"
      return 0
    fi
  fi
  echo "── [$name] 开始"
  ( "$@" ) 2>&1 | tee "$LOG_DIR/$name-$RUN_TAG.log"
  if [[ "$name" != deps && "$name" != env && "$name" != smoke ]]; then
    state mark --receipt "$STATE_DIR/$name.json" --inputs "${INPUTS[@]}" --outputs "${OUTPUTS[@]}"
  fi
  echo "── [$name] 完成"
}

stage_smoke() {
  # 在付费采集和长时间训练之前，把只能在 GPU 上验证的环节跑一遍：
  # NF4 量化、LoRA、一步训练、一次生成
  pyrun scripts/smoke_model.py --model "$MODEL" --revision "$MODEL_REVISION" \
    --quantization "$QUANTIZATION" --lora-r "${LORA_R:-16}" --lora-alpha "${LORA_ALPHA:-32}" \
    --max-length "${MAX_LENGTH:-2048}"
}

require_empty() {
  pyrun - "$1" "$2" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
if path.exists() and (not path.is_dir() or any(path.iterdir())):
    raise SystemExit(f'已有产物不会覆盖：{path}。请设置新的 {sys.argv[2]}；旧数据保留。')
PY
}

stage_deps() {
  local extra
  extra="$(state install-extra "$MODEL")"     # train（默认，含 Qwen3.5）| train-qwen25（Qwen2.5 系列）
  pyrun -m pip install -e ".[dev,${extra},api]"
  pyrun -m pip check
  pyrun -c 'import torch, transformers, peft, bitsandbytes, accelerate, openai; print("训练依赖导入成功")'
}

stage_env() {
  # 下载权重之前先确认架构受支持（Qwen3.5 需要 transformers >= 5.2.0），再用真实
  # tokenizer 渲染一次 chat template：transformers 5.x 要求 jinja2 >= 3.1.6，
  # 现成的环境里常常是旧版本，在这里报错比等到训练时才发现好
  pyrun - "$MODEL" "$MODEL_REVISION" <<'PY'
import sys
sys.path.insert(0, "src")
from agentlab.training import hf_runtime
model, revision = sys.argv[1], sys.argv[2] or None
hf_runtime.assert_model_supported(model, revision)
print(f"模型架构检查通过: {model} @ {revision or 'main'}")
try:
    tokenizer = hf_runtime.load_tokenizer(model, revision)
    tokens = hf_runtime.apply_chat_template(
        tokenizer, [{"role": "user", "content": "ping"}], add_generation_prompt=True)
except ImportError as error:
    raise SystemExit(f"chat template 无法渲染: {error}\n"
                     '在当前环境执行: python -m pip install -U "jinja2>=3.1.6"')
print(f"chat template 渲染正常（{len(tokens)} 个 token）")
PY
  # 判分隔离的实际情况：landlock 不可用时候选代码只在 rlimit + 清洁环境下运行
  pyrun -m agentlab doctor | grep -E '^sandbox\.'
  # 在沙箱里真实启动一次解释器。放行规则按当前解释器的安装方式计算，conda、独立
  # 构建或容器镜像里的 Python 可能缺库目录；在这里失败，好过后面每道题都判分失败
  pyrun - <<'PY'
import sys
sys.path.insert(0, "src")
from agentlab.runtime.isolation import self_test
passed, detail = self_test()
print(f"沙箱自检{'通过' if passed else '失败'}: {detail}")
raise SystemExit(0 if passed else 1)
PY
  pyrun - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA 不可用：检查驱动/环境'
assert torch.cuda.is_bf16_supported(), '需要 bf16 支持'
print('GPU:', torch.cuda.get_device_name(0))
print('显存 GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
tensor = torch.randn(64, 64, device='cuda', requires_grad=True)
tensor.square().mean().backward()
assert torch.isfinite(tensor.grad).all()
print('GPU forward/backward: OK')
PY
  PYTHONPATH=src pyrun -m pytest tests -q
}

stage_pools() {
  local export_dir="$POOL_DIR/$RUN_TAG"
  mkdir -p "$export_dir"
  pyrun scripts/export_pilot_tasks.py --output "$export_dir/pilot.jsonl"
  local pools files=("$export_dir/pilot.jsonl") pool
  pools="$(pyrun scripts/export_tasks_v2.py --list-pools)"
  for pool in $pools; do
    pyrun scripts/export_tasks_v2.py --pool "$pool" --output "$export_dir/$pool.jsonl"
    files+=("$export_dir/$pool.jsonl")
  done
  cat "${files[@]}" > "$POOL_DIR/combined.jsonl"
  echo "合并池任务数: $(wc -l < "$POOL_DIR/combined.jsonl")（池: pilot $pools）"
}

stage_split() {
  local stratify=()
  [[ "$SPLIT_STRATIFY" != 1 ]] || stratify=(--stratify)
  if [[ -f "$EXP_DIR/tasks.jsonl" && -f "$EXP_DIR/split-manifest.json" && -f "$EXP_DIR/audit-manifest.json" ]]; then
    state validate-split --directory "$EXP_DIR" --pool "$POOL_DIR/combined.jsonl" \
      --seed "$SEED" --dev-ratio "$DEV_RATIO" --test-ratio "$TEST_RATIO" "${stratify[@]}"
    echo "已核对冻结任务与当前池一致：$EXP_DIR"
    return 0
  fi
  require_empty "$EXP_DIR" EXP_DIR
  pyrun scripts/prepare_data.py --tasks-file "$POOL_DIR/combined.jsonl" --split-tasks \
    --output-dir "$EXP_DIR" --dev-ratio "$DEV_RATIO" --test-ratio "$TEST_RATIO" --seed "$SEED" "${stratify[@]}"
}

collection_status() {
  local split="$1" output="$2" count="$3"
  pyrun scripts/collect_data.py --tasks-file "$EXP_DIR/tasks.jsonl" --split "$split" \
    --candidates "$count" --output "$output" --status-only
}

stage_collect() {
  local train_status=0 dev_status=0
  collection_status train "$TRAIN_CANDIDATES_FILE" "$TRAIN_CANDIDATES" || train_status=$?
  collection_status dev "$DEV_CANDIDATES_FILE" "$DEV_CANDIDATES" || dev_status=$?
  if [[ "$train_status" != 0 && "$train_status" != 3 ]] || [[ "$dev_status" != 0 && "$dev_status" != 3 ]]; then
    echo '采集清单或协议不匹配，请修复路径或使用新的候选输出路径。' >&2
    return 2
  fi
  if [[ "$train_status" == 0 && "$dev_status" == 0 ]]; then
    echo '所有唯一 task/candidate 槽位均完整，无需 API key。'
    return 0
  fi
  [[ -n "${TEACHER_API_KEY:-}" ]] || { echo '缺少 TEACHER_API_KEY，请在环境变量或 .env 中设置。' >&2; return 1; }
  echo "按冻结 train/dev 采集，train 每题 $TRAIN_CANDIDATES，dev 每题 $DEV_CANDIDATES（付费请求）。"
  echo '上面两行 status 的 missing 数量即将开始付费采集；核对 slots/missing 无误后放行。'
  if [[ "${ASSUME_YES:-0}" != 1 ]]; then
    # 网页终端无法向管道内脚本传递交互输入（read 会读到 EOF 自动“取消”），
    # 改用倒计时提供取消窗口。
    local second
    for second in 10 9 8 7 6 5 4 3 2 1; do
      printf '\r  %d 秒后开始付费采集，Ctrl+C 取消……' "$second"
      sleep 1
    done
    printf '\n  开始付费采集。\n'
  fi
  local retry=()
  [[ "${RETRY_ERRORS:-0}" != 1 ]] || retry=(--retry-errors)
  if [[ "$train_status" != 0 ]]; then
    pyrun scripts/collect_data.py --tasks-file "$EXP_DIR/tasks.jsonl" --split train \
      --candidates "$TRAIN_CANDIDATES" --output "$TRAIN_CANDIDATES_FILE" "${retry[@]}"
  fi
  if [[ "$dev_status" != 0 ]]; then
    pyrun scripts/collect_data.py --tasks-file "$EXP_DIR/tasks.jsonl" --split dev \
      --candidates "$DEV_CANDIDATES" --output "$DEV_CANDIDATES_FILE" "${retry[@]}"
  fi
  collection_status train "$TRAIN_CANDIDATES_FILE" "$TRAIN_CANDIDATES"
  collection_status dev "$DEV_CANDIDATES_FILE" "$DEV_CANDIDATES"
}

stage_prepare() {
  require_empty "$SFT_DIR" SFT_DIR
  local stale=()
  [[ "${REVALIDATE_STALE:-0}" != 1 ]] || stale=(--revalidate-stale)
  pyrun scripts/prepare_data.py --input "$TRAIN_CANDIDATES_FILE" "$DEV_CANDIDATES_FILE" \
    --tasks-file "$EXP_DIR/tasks.jsonl" --output-dir "$SFT_DIR" --max-per-task "$MAX_PER_TASK" "${stale[@]}"
}

stage_train() {
  require_empty "$CKPT_DIR" CKPT_DIR
  pyrun scripts/train_sft.py --data "$SFT_DIR/train.jsonl" --eval-data "$SFT_DIR/dev.jsonl" \
    --model "$MODEL" --revision "$MODEL_REVISION" --output "$CKPT_DIR" "${TRAIN_ARGS[@]}" --dry-run
  pyrun scripts/train_sft.py --data "$SFT_DIR/train.jsonl" --eval-data "$SFT_DIR/dev.jsonl" \
    --model "$MODEL" --revision "$MODEL_REVISION" --output "$CKPT_DIR" "${TRAIN_ARGS[@]}"
}

validate_checkpoint() {
  pyrun - "$CKPT_DIR/training-manifest.json" "$MODEL" "$MODEL_REVISION" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding='utf-8'))
if manifest.get('status', 'complete') != 'complete':
    raise SystemExit('训练未完成；请选择完整 checkpoint 或新 CKPT_DIR 重训。')
if manifest['model'] != sys.argv[2] or sys.argv[3] not in {manifest.get('revision'), manifest.get('resolved_revision')}:
    raise SystemExit('checkpoint 模型/revision 与配置不一致；请显式设置 MODEL/MODEL_REVISION。')
PY
}

stage_select() {
  validate_checkpoint
  require_empty "$SELECTION_DIR" EVAL_OUTPUT_DIR
  pyrun scripts/select_checkpoint.py --checkpoint-root "$CKPT_DIR" --tasks-file "$EXP_DIR/tasks.jsonl" \
    --output "$SELECTION_DIR" --quantization "$QUANTIZATION" --max-new-tokens "$MAX_NEW_TOKENS" --seed "$SEED" \
    "${EVAL_FLAGS[@]}"
}

stage_eval() {
  local split="$1" output="$EVAL_OUTPUT_DIR/$1"
  validate_checkpoint
  artifacts select
  state check --receipt "$STATE_DIR/select.json" --inputs "${INPUTS[@]}" --outputs "${OUTPUTS[@]}" || {
    echo '先完成当前配置的 select(dev) 阶段；已有选择已失效时请使用新 EVAL_OUTPUT_DIR。' >&2; return 1;
  }
  require_empty "$output" EVAL_OUTPUT_DIR
  local adapter
  adapter="$(pyrun - "$SELECTION_DIR/selection.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))['selected_adapter'] or '')
PY
)"
  local adapter_args=()
  if [[ -n "$adapter" ]]; then
    adapter_args=(--adapter "$adapter")
  else
    echo 'dev 未发现优于基线的 checkpoint，保留 base；这不代表 SFT 已获得提升。'
  fi
  mkdir -p "$output"
  if [[ "$split" == dev ]]; then
    pyrun - "$SELECTION_DIR" "$output" <<'PY'
import json, shutil, sys
from pathlib import Path
source, destination = map(Path, sys.argv[1:])
selected = json.loads((source / 'selection.json').read_text(encoding='utf-8'))
shutil.copyfile(source / 'epoch-000-dev.json', destination / 'base.json')
shutil.copyfile(source / f"epoch-{selected['selected_epoch']:03d}-dev.json", destination / 'selected.json')
PY
  else
    pyrun scripts/evaluate.py --tasks-file "$EXP_DIR/tasks.jsonl" --split "$split" \
      --model "$MODEL" --revision "$MODEL_REVISION" --training-data "$CKPT_DIR/training-manifest.json" \
      --quantization "$QUANTIZATION" --temperature 0 --seed "$SEED" --max-new-tokens "$MAX_NEW_TOKENS" \
      --output "$output/base.json" "${EVAL_FLAGS[@]}"
    if [[ -n "$adapter" ]]; then
      pyrun scripts/evaluate.py --tasks-file "$EXP_DIR/tasks.jsonl" --split "$split" \
        --model "$MODEL" --revision "$MODEL_REVISION" "${adapter_args[@]}" --training-data "$CKPT_DIR/training-manifest.json" \
        --quantization "$QUANTIZATION" --temperature 0 --seed "$SEED" --max-new-tokens "$MAX_NEW_TOKENS" \
        --output "$output/selected.json" "${EVAL_FLAGS[@]}"
    else
      cp "$output/base.json" "$output/selected.json"
    fi
  fi
  pyrun scripts/compare.py --baseline "$output/base.json" --treatment "$output/selected.json" \
    --bootstrap-samples 10000 --seed "$SEED" --output "$output/comparison.md"
  cat "$output/comparison.md"
}

if [[ "$EVAL_SPLIT" != dev && "$STAGE" != final-eval ]]; then
  echo 'eval/all 只允许 dev；最终 test 请显式运行 final-eval。' >&2
  exit 2
fi
case "$STAGE" in
  env) run_guarded env stage_env ;;
  smoke) run_guarded smoke stage_smoke ;;
  deps) run_guarded deps stage_deps ;;
  pools) run_guarded pools stage_pools ;;
  split) run_guarded split stage_split ;;
  collect) run_guarded collect stage_collect ;;
  prepare) run_guarded prepare stage_prepare ;;
  train) run_guarded train stage_train ;;
  select) run_guarded select stage_select ;;
  eval) run_guarded eval stage_eval dev ;;
  final-eval) run_guarded final-eval stage_eval test ;;
  all)
    run_guarded deps stage_deps
    run_guarded env stage_env
    run_guarded smoke stage_smoke
    run_guarded pools stage_pools
    run_guarded split stage_split
    run_guarded collect stage_collect
    run_guarded prepare stage_prepare
    run_guarded train stage_train
    run_guarded select stage_select
    run_guarded eval stage_eval dev
    echo "dev 流程完成。选择结果：$SELECTION_DIR/selection.json"
    echo '锁定配置后，保持同样环境变量，单独运行 final-eval 评估 test。'
    ;;
  *) echo '未知阶段：可选 deps env smoke pools split collect prepare train select eval final-eval all' >&2; exit 2 ;;
esac
