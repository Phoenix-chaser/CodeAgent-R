#!/usr/bin/env bash
# 阶梯上的 RFT 第 1 轮：让基座模型对难题和单缺陷变体各采样若干次并逐个判分，
# 再用同一批采样训练三组对照，全部只在 dev 上选型。test 这一轮不评。
#   A 验证组：只留判分通过的答案，lr 1e-4、LoRA 秩 32（CodeWalk 的配方，arXiv:2610.15345）
#   C 对照组：同样的题，每题随机取一份不检查对错的答案，超参同 A
#   B 旧超参：数据同 A，lr 1e-5、秩 16（此前 RFT 用的设置）
# 采样走本机 vLLM 服务（脚本自己启动、用完关闭），训练和评测走现有环境。
# 每一步的产物已存在就跳过，中断后重新启动会接着跑。看进度：
#   WATCH=30 bash scripts/show_progress.sh <日志> "" run_rft.sh
# 可选的开关（环境变量）：
#   STAGES=sampling        采样那一步做完就关掉 vLLM 并停下，训练另找时间做（题多到一晚跑不完时拆成两晚：
#                          第二晚去掉这个变量再运行，采样已经做完，会直接跳到准备数据）
#   SKIP_ARMS="B-oldhp"    空格分隔的组名，这些组不训；最后的汇总和 vLLM 评测也只列没跳过的组
#   VLLM_EVAL=1            汇总之后经本机 vLLM 服务把没跳过的组再评一遍（辅助指标，见 scripts/rft_vllm_eval.sh）
# 采样中途 vLLM 退出（显存不足、被系统杀掉）时脚本把它重新启动一次再接着采，每个 split 最多重启一次。
# 结束时：要做的都做完了才是「===== 全部结束 =====」，退出码 0；vLLM 评测没评完（它是辅助指标，不影响 dev 上的
# 选型结果）时最后一行是「===== 没有全部完成：… =====」，退出码 1，重新运行同一条命令只补没做完的。
cd "$(dirname "$0")/.."
unset AGENTLAB_SANDBOX
export PYTHONPATH=$PWD/src PYTHONUNBUFFERED=1 HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
state() { python src/agentlab/experiments/run_state.py "$@"; }
MODEL="${MODEL:-$(state default-model)}"
REVISION="${MODEL_REVISION:-$(state revision "$MODEL")}"
TASKS="${TASKS_FILE:-data/frozen-ladder.jsonl}"
RUN="${RUN_DIR:-data/rft-round1}"
TIERS="${RFT_TIERS:-hard,variant}"
SAMPLES="${RFT_SAMPLES:-8}"
EPOCHS="${RFT_EPOCHS:-1}"
NEW_LR="${RFT_LR:-1e-4}" NEW_R="${RFT_LORA_R:-32}" NEW_ALPHA="${RFT_LORA_ALPHA:-64}"
OLD_LR="${OLD_LR:-1e-5}" OLD_R="${OLD_LORA_R:-16}" OLD_ALPHA="${OLD_LORA_ALPHA:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-8}" MAX_LENGTH="${MAX_LENGTH:-2048}"
VLLM_BIN="${VLLM_BIN:-$HOME/vllm-qwen/bin/vllm}"
VLLM_PORT="${VLLM_PORT:-8000}"
URL="http://127.0.0.1:$VLLM_PORT"
CONCURRENCY="${RFT_CONCURRENCY:-24}" GRADE_WORKERS="${RFT_GRADE_WORKERS:-4}"
POLL="${RFT_POLL_SECONDS:-10}"       # 等 vLLM 启动时查看日志的间隔（秒），总共最多等 15 分钟
ATTEMPTS=$(( 900 / POLL )); (( ATTEMPTS >= 1 )) || ATTEMPTS=1
BASE_REPORT="${BASE_REPORT:-}"       # 可选：已有的基座 dev 报告，用来省掉重评基座
TOTAL=7
STARTED=$SECONDS
STAGES="${STAGES:-all}"                # sampling：采样做完就停
SKIP_ARMS="${SKIP_ARMS:-}"             # 空格分隔的组名，这些组不训
VLLM_EVAL="${VLLM_EVAL:-0}"            # 1：汇总之后经本机 vLLM 服务评没跳过的组（辅助指标）
[[ "$STAGES" == all || "$STAGES" == sampling ]] \
  || { echo "STAGES 只认 all（默认，整轮做完）或 sampling（采样做完就停），现在是 $STAGES"; exit 1; }
[[ "$VLLM_EVAL" == 0 || "$VLLM_EVAL" == 1 ]] || { echo "VLLM_EVAL 只能是 0 或 1（现在是 $VLLM_EVAL）"; exit 1; }
mkdir -p "$RUN/checkpoints" "$RUN/results"

BAR="########################" DASH="------------------------"
progress() {                          # 已完成 总数 说明
  local filled=$(( 24 * $1 / $2 ))
  local minutes=$(( (SECONDS - STARTED) / 60 ))
  echo "[${BAR:0:filled}${DASH:0:$((24 - filled))}] $1/$2 ($(( $1 * 100 / $2 ))%) $3 · 已用 ${minutes}m"
}
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' '; }
stop_vllm() { pkill -f 'vllm serve' 2>/dev/null; pkill -f 'VLLM::' 2>/dev/null; }
vllm_running() {                     # 自己启动的 vLLM 还在跑（没退出，也不是已经退出还没收回的僵尸）
  [[ -n "${VLLM_PID:-}" ]] || return 1
  kill -0 "$VLLM_PID" 2>/dev/null || return 1
  # /proc/<pid>/stat 是 "pid (进程名) 状态 …"，进程名里可能有空格和括号：取最后一个 ") " 后面的那个字母
  local state
  state="$(</proc/"$VLLM_PID"/stat)" || return 1
  [[ "${state##*) }" != Z* ]]
}
sampling_complete() {                # train 或 dev：这个 split 的采样做完了（清单在，而且槽数 = 采样数 = 判分数）
  # 清单在不等于做完：旧版本的 sample_rft.py 读不回某些采样时照样写一份少的清单。不是这样就没做完，原样重跑只补
  python - "$RUN/sampling/$1-manifest.json" <<'PY'
import json, sys
from pathlib import Path
try:
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    complete = manifest["slots"] == manifest["raw_records"] == manifest["graded_records"] > 0
except (OSError, ValueError, KeyError, TypeError):
    complete = False
sys.exit(0 if complete else 1)
PY
}
start_vllm() {                       # 启动 vLLM 并等它就绪；起不来返回非 0（原因和日志末尾自己会打印）
  stop_vllm; sleep 3                # 上一次脚本中途断掉时留下的服务会占着端口，先清掉再等它退出
  nohup "$VLLM_BIN" serve "$MODEL" --revision "$REVISION" --language-model-only --max-model-len 8192 \
    --port "$VLLM_PORT" > "$RUN/vllm.log" 2>&1 &
  VLLM_PID=$!
  local i ready=0
  for i in $(seq 1 "$ATTEMPTS"); do
    if grep -q 'Application startup complete' "$RUN/vllm.log" 2>/dev/null; then ready=1; break; fi
    if grep -q 'Engine core initialization failed' "$RUN/vllm.log" 2>/dev/null; then break; fi
    kill -0 "$VLLM_PID" 2>/dev/null || break
    (( i % 3 != 0 )) || progress "$i" "$ATTEMPTS" "等待 vLLM 加载"
    sleep "$POLL"
  done
  if [[ "$ready" != 1 ]]; then
    echo "vLLM 没起来，日志最后 15 行："; tail -n 15 "$RUN/vllm.log" | cut -c1-200
    stop_vllm; return 1
  fi
  progress "$ATTEMPTS" "$ATTEMPTS" "vLLM 已就绪"
}
wait_gpu_free() {                    # vLLM 收到关闭信号后，引擎子进程还要几秒到几十秒才放掉显存；最多等两分钟
  local i used
  for i in $(seq 1 24); do used="$(gpu_used)"; [[ -n "${used:-}" && $(( used > 2000 )) == 1 ]] || return 0; sleep 5; done
  used="$(gpu_used)"
  echo "vLLM 关掉两分钟后显卡仍被占用（已用 ${used} MiB），训练会显存不足，先停在这里。占着显卡的进程："
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | head -n 5
  return 1
}

echo "===== 1/$TOTAL 检查前提 ====="
[[ -s "$TASKS" ]] || { echo "缺少任务文件 $TASKS：先把冻结的阶梯划分拷到这个路径"; exit 1; }
[[ -n "$MODEL" && -n "$REVISION" ]] \
  || { echo "查不到默认模型或固定版本（$MODEL / $REVISION），把这一行贴给我"; exit 1; }
[[ -x "$VLLM_BIN" ]] || { echo "找不到 vLLM：$VLLM_BIN（用 VLLM_BIN 指定它的路径）"; exit 1; }
python - "$TASKS" <<'PY'
import collections, json, sys
specs = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
counts = collections.Counter(spec["split"] for spec in specs)
ladder = collections.Counter(spec["split"] for spec in specs if spec["task_id"].startswith(("h3_", "h3v_")))
print(f"任务文件：{len(specs)} 题，train {counts['train']} / dev {counts['dev']} / test {counts['test']}；"
      f"其中难题来源族 train {ladder['train']} / dev {ladder['dev']}")
synthetic = collections.Counter(spec["split"] for spec in specs if spec["task_id"].startswith(("s3_", "s3v_")))
if sum(synthetic.values()):    # 这一批里没有合成题时不提
    print(f"其中合成题（s3_、s3v_ 开头）train {synthetic['train']} / dev {synthetic['dev']} / test {synthetic['test']}")
PY
used="$(gpu_used)"
if [[ -n "${used:-}" ]] && (( used > 2000 )); then
  echo "显卡被占用（已用 ${used} MiB）。先停掉占显卡的程序（例如上一轮的 vllm serve 或评测）"
  echo "再启动："
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | head -n 5
  exit 1
fi

if sampling_complete train && sampling_complete dev; then
  echo "===== 2/$TOTAL 采样已完成，跳过 ====="
else
  echo "===== 2/$TOTAL 启动 vLLM 并采样（难题和变体，每题 $SAMPLES 次）====="
  start_vllm || exit 1
  for split in train dev; do
    if sampling_complete "$split"; then echo "$split 的采样已经做完了，跳过（只补另一个 split 缺的槽位）"; continue; fi
    ok=0 restarts=0
    for attempt in 1 2 3; do
      rc=0
      python scripts/sample_rft.py --tasks-file "$TASKS" --split "$split" --tiers "$TIERS" \
        --samples "$SAMPLES" --vllm-url "$URL" --model "$MODEL" --concurrency "$CONCURRENCY" \
        --grade-workers "$GRADE_WORKERS" --output-dir "$RUN/sampling" || rc=$?
      if (( rc == 0 )); then ok=1; break; fi
      if (( rc == 2 )); then
        # 2 是参数或采样协议不对（两次运行之间改了采样次数之类）：重试多少次都一样，服务其实没有问题
        echo "采样 $split 没有做成：sample_rft.py 以退出码 2 结束，是参数或采样协议不对，重试没有用。把上面的报错贴给我"
        stop_vllm; exit 1
      fi
      if ! vllm_running && (( attempt < 3 )); then
        # 没跑完的时候 vLLM 已经不在了（显存不足、被系统杀掉）：对着一个不在的服务重试没有用，重新启动一次再试
        if (( restarts )); then
          echo "采样 $split 没跑完，vLLM 又不在了（已经重新启动过一次），停止。vLLM 日志最后 15 行："
          tail -n 15 "$RUN/vllm.log" | cut -c1-200
          stop_vllm; exit 1
        fi
        restarts=1
        echo "采样 $split 没跑完的时候 vLLM 已经不在了。vLLM 日志最后 15 行："
        tail -n 15 "$RUN/vllm.log" | cut -c1-200
        echo "重新启动 vLLM 后再试（只补没完成的槽位）"
        stop_vllm                  # 引擎的子进程可能还占着显存：先清掉，等它放出来再启动
        wait_gpu_free || exit 1
        start_vllm || exit 1
        continue
      fi
      echo "采样 $split 第 $attempt 次没跑完，5 秒后重试（只补没完成的槽位）"; sleep 5
    done
    if [[ "$ok" != 1 ]]; then
      echo "采样 $split 三次都没完成，停止。vLLM 日志最后 15 行："
      tail -n 15 "$RUN/vllm.log" | cut -c1-200
      stop_vllm; exit 1
    fi
  done
  echo "采样结束，关闭 vLLM"
  stop_vllm
  wait_gpu_free || exit 1
fi
if [[ "$STAGES" == sampling ]]; then
  # 不打印"全部结束"：这一轮还没做完，进度面板会如实显示没有在运行
  echo "结果：采样阶段结束"
  echo "（STAGES=sampling：到这里就停。训练时去掉这个变量再运行一次，采样已经做完，会直接跳到准备数据）"
  exit 0
fi

echo "===== 3/$TOTAL 准备训练数据 ====="
if [[ ! -s "$RUN/sft/verified/dataset-manifest.json" ]]; then
  python scripts/prepare_data.py --tasks-file "$TASKS" \
    --input "$RUN/sampling/train-candidates.jsonl" "$RUN/sampling/dev-candidates.jsonl" \
    --output-dir "$RUN/sft/verified" --max-per-task 1 \
    || { echo "准备验证组数据失败。如果是因为没有任何样本通过，这一轮就没有可训练的数据。"
         exit 1; }
fi
python - "$RUN/sft/verified" <<'PY'
import collections, json, sys
from pathlib import Path
root = Path(sys.argv[1])
for split in ("train", "dev"):
    samples = [json.loads(line) for line in (root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    # 同 src/agentlab/benchmark/tiers.py 的定义（这里只用标准库，不去导入它）；先判 s3v_ 再判 s3_
    labels = {"h3_": "难题", "h3v_": "变体", "s3v_": "合成变体", "s3_": "合成难题"}
    tiers = collections.Counter(next((label for prefix, label in labels.items() if s["task_id"].startswith(prefix)),
                                     "原有题") for s in samples)
    tasks = len({sample["task_id"] for sample in samples})
    print(f"结果：验证组 {split} {len(samples)} 条，覆盖 {tasks} 题；分层 {dict(tiers)}")
PY
if [[ ! -s "$RUN/sft/control/dataset-manifest.json" ]]; then
  python scripts/rft_control.py --graded "$RUN/sampling/train-graded.jsonl" \
    --verified-dir "$RUN/sft/verified" --output-dir "$RUN/sft/control" --seed 42 || exit 1
fi

training_complete() {
  python - "$1" <<'PY'
import json, sys
from pathlib import Path
try:
    complete = json.load(open(Path(sys.argv[1]) / "training-manifest.json")).get("status") == "complete"
except (OSError, ValueError):
    complete = False
sys.exit(0 if complete else 1)
PY
}

# 这一组是不是列在 SKIP_ARMS 里（空格分隔的组名）
skipped() {
  local name
  for name in $SKIP_ARMS; do [[ "$name" == "$1" ]] && return 0; done
  return 1
}

# 训练一组并在 dev 上选型。参数依次是：名称、数据目录、学习率、LoRA 秩、LoRA alpha、步序号；
# 多出来的参数原样传给训练脚本
arm() {
  local name="$1" data="$2" lr="$3" rank="$4" alpha="$5" step="$6"; shift 6
  if skipped "$name"; then
    echo "===== $step/$TOTAL $name：列在 SKIP_ARMS 里，这一组不训，跳过 ====="
    return
  fi
  local ckpt="$RUN/checkpoints/$name" select="$RUN/results/$name"
  if [[ -s "$select/selection.json" ]]; then
    echo "===== $step/$TOTAL $name：已经选过型，跳过 ====="
  else
    echo "===== $step/$TOTAL $name：训练（lr $lr、秩 $rank）并在 dev 上选型 ====="
    if ! training_complete "$ckpt"; then
      if [[ -d "$ckpt" ]]; then
        local moved="$ckpt.incomplete-$(date +%H%M%S)"
        mv "$ckpt" "$moved"; echo "上次的训练没有完成，目录移到 $moved，重新训练"
      fi
      local train_args=(--data "$data/train.jsonl" --eval-data "$data/dev.jsonl" --model "$MODEL"
        --revision "$REVISION" --output "$ckpt" --epochs "$EPOCHS" --lr "$lr" --lora-r "$rank"
        --lora-alpha "$alpha" --batch-size 1 --grad-accum "$GRAD_ACCUM" --max-length "$MAX_LENGTH"
        --seed 42 "$@")
      python scripts/train_sft.py "${train_args[@]}" --dry-run \
        || { echo "训练数据没通过检查，把上面的报错贴给我"; exit 1; }
      python scripts/train_sft.py "${train_args[@]}" \
        || { echo "训练出错，把上面的报错贴给我"; exit 1; }
    else
      echo "训练已经完成过，跳过训练"
    fi
    local flags=(--resume)
    [[ -z "$BASE_REPORT" ]] || flags+=(--base-report "$BASE_REPORT")
    python scripts/select_checkpoint.py --checkpoint-root "$ckpt" --tasks-file "$TASKS" \
      --output "$select" --quantization nf4 --max-new-tokens 2048 --seed 42 "${flags[@]}" \
      || { echo "选型出错，把上面的报错贴给我"; exit 1; }
  fi
  [[ -n "$BASE_REPORT" ]] || BASE_REPORT="$select/epoch-000-dev.json"
  python scripts/tier_report.py "$select" || exit 1
  python - "$select/selection.json" "$name" <<'PY'
import json, sys
selection = json.load(open(sys.argv[1]))
chosen = next(row for row in selection["candidates"] if row["epoch"] == selection["selected_epoch"])
what = "保留基座" if selection["selected_epoch"] == 0 else f"选中 epoch {selection['selected_epoch']}"
print(f"结果：{sys.argv[2]} dev {selection['base_pass_at_1']:.3f} → {chosen['pass_at_1']:.3f}"
      f"（{what}；变好 {chosen['improved_tasks']} 道、变差 {chosen['regressed_tasks']} 道）")
PY
}
arm A-verified "$RUN/sft/verified" "$NEW_LR" "$NEW_R" "$NEW_ALPHA" 4
arm C-control "$RUN/sft/control" "$NEW_LR" "$NEW_R" "$NEW_ALPHA" 5 --allow-unverified
arm B-oldhp "$RUN/sft/verified" "$OLD_LR" "$OLD_R" "$OLD_ALPHA" 6

echo "===== 7/$TOTAL 汇总（都在 dev 上；test 这一轮不评）====="
# 只列没跳过、而且已经选完的组
kept=()
report_args=()
for name in A-verified C-control B-oldhp; do
  skipped "$name" && continue
  kept+=("$name")
  if [[ -s "$RUN/results/$name/selection.json" ]]; then
    report_args+=("$RUN/results/$name")
  else
    echo "results/$name：还没有选型结果，不列进汇总"       # 不列的组要说一声：表里没有它的行，按行数会把组认错
  fi
done
if (( ${#report_args[@]} )); then
  python scripts/tier_report.py "${report_args[@]}" || echo "有的组还没选完，汇总只包含已经完成的。"
else
  echo "没有已经选完的组，汇总跳过。"
fi
incomplete=""                     # 没做完的事：有就不是"全部结束"
if [[ "$VLLM_EVAL" == 1 ]]; then
  # 辅助指标：没评完不影响上面的 dev 选型结果，但这一轮不能算"全部结束"——rft_vllm_eval.sh 自己不打印"全部结束"、
  # 以 1 退出，这里不能替它打印再以 0 退出（那样进度面板会显示"已完成"）。第 1 轮没有族内探针，也不对哪一组自检
  if (( ${#kept[@]} )); then
    echo "----- 经本机 vLLM 服务评没跳过的组选中的适配器（辅助指标）-----"
    rc=0
    # 第 1 轮没有族内探针，也不对哪一组做自检；INCLUDE_BASE 是 rft_vllm_eval.sh 自己的开关，照常从环境里读
    EXPECT_REPRODUCE="" PROBE_IDS_FILE="" bash scripts/rft_vllm_eval.sh "$RUN" "${kept[@]}" || rc=$?
    (( rc == 0 )) || incomplete="vLLM 评测没有全部完成（rft_vllm_eval.sh 以退出码 $rc 结束）"
  else
    echo "三组都列在 SKIP_ARMS 里，没有要评的组，vLLM 评测跳过。"
  fi
fi
if [[ -n "$incomplete" ]]; then
  echo "===== 没有全部完成：$incomplete；上面的 dev 选型结果不受影响，重新运行同一条命令只补没做完的 ====="
  exit 1
fi
echo "===== 全部结束 ====="
