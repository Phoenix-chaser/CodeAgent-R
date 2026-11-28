#!/usr/bin/env bash
# RFT 各个一键脚本共用的函数：进度条、查看显卡、启停 vLLM、训练一组并在 dev 上选型、vLLM 评测。
# 这个文件只定义函数和变量，不做别的事，用的时候 source：
#   cd "$(dirname "$0")/.."                # 先回到仓库根
#   source scripts/rft_lib.sh
# 调用方先设好要用到的变量（只用到的那些）：
#   RUN 运行目录；MODEL 基座模型名；REVISION 基座的固定版本；TASKS 冻结的任务文件；
#   VLLM_BIN vLLM 的可执行文件；VLLM_PORT 服务端口（默认 8000）；GRADE_WORKERS 同时判分的进程数（默认 4）；
#   EPOCHS、GRAD_ACCUM、MAX_LENGTH 训练参数；BASE_REPORT 已有的基座 dev 报告（可空，rft_arm 会改写它）；
#   TOTAL 步骤总数、STARTED 脚本开始时的 $SECONDS（步骤标记和进度条用）。
# 可选：RFT_POLL_SECONDS 等 vLLM 启动时查看日志的间隔（秒，默认 10）。
# 函数（参数写在各自的注释里）：
#   rft_progress                进度条，格式和 show_progress.sh 认的一样
#   rft_gpu_used / rft_gpu_processes / rft_check_gpu_free / rft_wait_gpu_free   查显存、等显存
#   rft_stop_vllm / rft_start_vllm / rft_vllm_running                           关服务、起服务、看自己起的还在不在
#   rft_training_complete / rft_sampling_complete / rft_sampling_lost / rft_arm
#                                训过没有；采样做完了没有（槽数 = 采样数 = 判分数）；清单里丢了几个槽位；训一组并在 dev 上选型
#   rft_eval_done / rft_eval_unconfirmed / rft_eval_dev / rft_eval_probe / rft_eval_vllm   经 vLLM 评测
# 出错时只有 rft_arm 直接 exit 1（和 run_rft.sh 原来的 arm() 一样）；其余函数都只返回非 0，由调用方决定。

RFT_BAR="########################" RFT_DASH="------------------------"

rft_progress() {                      # 已完成 总数 说明
  (( $# >= 2 && $2 > 0 )) || return 0
  local filled=$(( 24 * $1 / $2 ))
  local minutes=$(( (SECONDS - ${STARTED:-0}) / 60 ))
  echo "[${RFT_BAR:0:filled}${RFT_DASH:0:$((24 - filled))}] $1/$2 ($(( $1 * 100 / $2 ))%) $3 · 已用 ${minutes}m"
}

rft_gpu_used() {
  # 打印第一块显卡已用的显存（MiB）。只认纯数字：驱动异常时 nvidia-smi 会写出 [N/A] 之类，
  # 拿去做算术比较会报错，这时当作查不到（打印空）
  local used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]')"
  [[ "$used" =~ ^[0-9]+$ ]] || return 0
  echo "$used"
}

rft_gpu_processes() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | head -n 5; }

rft_check_gpu_free() {
  # 一开始就查：上一轮的 vllm serve 或评测还占着显卡时，训练和新的服务都会显存不足
  local used
  used="$(rft_gpu_used)"
  [[ -n "$used" && "$used" -gt 2000 ]] || return 0
  echo "显卡被占用（已用 ${used} MiB）。先停掉占显卡的程序（多半是上一轮的 vllm serve 或评测）再运行。占着显卡的进程："
  rft_gpu_processes
  return 1
}

rft_wait_gpu_free() {
  # vLLM 收到关闭信号后，引擎子进程还要几秒到几十秒才放掉显存；最多等两分钟
  local i used
  for i in $(seq 1 24); do
    used="$(rft_gpu_used)"
    [[ -n "$used" && "$used" -gt 2000 ]] || return 0
    (( i % 6 != 0 )) || rft_progress "$i" 24 "等待显存释放"
    sleep 5
  done
  used="$(rft_gpu_used)"
  echo "vLLM 关掉两分钟后显卡仍被占用（已用 ${used} MiB），接着训练会显存不足，先停在这里。占着显卡的进程："
  rft_gpu_processes
  return 1
}

rft_stop_vllm() {
  # 先关这次自己启动的那个（rft_start_vllm 记下的 PID），再按进程名兜底：清掉上一次脚本中途断掉时
  # 留下的服务，以及服务退出后还占着显存的引擎子进程（vLLM 把它们的名字设成 VLLM:: 开头）
  [[ -z "${RFT_VLLM_PID:-}" ]] || { kill "$RFT_VLLM_PID" 2>/dev/null; unset RFT_VLLM_PID; }
  pkill -f 'vllm serve' 2>/dev/null
  pkill -f 'VLLM::' 2>/dev/null
  return 0
}

rft_vllm_running() {                    # rft_start_vllm 记下的那个 vLLM 还在跑（没退出，也不是已经退出还没收回的僵尸）返回 0
  [[ -n "${RFT_VLLM_PID:-}" ]] || return 1
  kill -0 "$RFT_VLLM_PID" 2>/dev/null || return 1
  [[ -d /proc/self ]] || return 0     # 没有 /proc 的系统分不出僵尸：进程还在就算在
  # /proc/<pid>/stat 是 "pid (进程名) 状态 …"，进程名里可能有空格和括号：取最后一个 ") " 后面的那个字母
  local state
  state="$(</proc/"$RFT_VLLM_PID"/stat)" || return 1
  [[ "${state##*) }" != Z* ]]
}

rft_start_vllm() {                     # 日志文件 [vllm serve 的额外参数…]；启动成功返回 0
  local log="$1"; shift
  local poll="${RFT_POLL_SECONDS:-10}"
  [[ "$poll" =~ ^[1-9][0-9]*$ ]] || poll=10        # 间隔不是正整数时用默认值，免得算次数时除零
  local attempts=$(( 900 / poll )); (( attempts >= 1 )) || attempts=1
  [[ -x "$VLLM_BIN" ]] || { echo "找不到 vLLM：$VLLM_BIN（用 VLLM_BIN 指定它的路径）"; return 1; }
  rft_stop_vllm; sleep 3                # 上一次脚本中途断掉时留下的服务会占着端口，先清掉再等它退出
  mkdir -p "$(dirname "$log")"
  nohup "$VLLM_BIN" serve "$MODEL" --revision "$REVISION" --language-model-only --max-model-len 8192 \
    --port "$VLLM_PORT" "$@" > "$log" 2>&1 < /dev/null &
  RFT_VLLM_PID=$!
  local i ready=0 reason="等了 15 分钟还没就绪"
  for i in $(seq 1 "$attempts"); do
    if grep -q 'Application startup complete' "$log" 2>/dev/null; then ready=1; break; fi
    if grep -q 'Engine core initialization failed' "$log" 2>/dev/null; then reason="引擎初始化失败"; break; fi
    # 进程可能刚好看完日志、还没轮到查它就写完退出了：日志里写明了原因就说具体的
    if ! kill -0 "$RFT_VLLM_PID" 2>/dev/null; then
      if grep -q 'Engine core initialization failed' "$log" 2>/dev/null; then reason="引擎初始化失败"; else reason="进程在就绪之前就退出了"; fi
      break
    fi
    (( i % 3 != 0 )) || rft_progress "$i" "$attempts" "等待 vLLM 加载"
    sleep "$poll"
  done
  if (( ! ready )); then
    echo "vLLM 没起来（$reason），日志最后 15 行："
    tail -n 15 "$log" | cut -c1-200
    rft_stop_vllm
    return 1
  fi
  rft_progress "$attempts" "$attempts" "vLLM 已就绪"
  return 0
}

rft_sampling_complete() {              # 采样目录 <split>[-<标记>]-manifest.json：这一份采样做完了返回 0
  # 清单在不算，还要 槽数 = 读得回的采样数 = 判分数：sample_rft.py 只在每一个槽位都有记录时才写清单，旧版本
  # 却会在读不回某些采样时照样写一份少的（那几道题就被记成"一次都没通过"）。不是这样就没做完，原样重跑只补
  python - "$1/$2" <<'PY'
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

rft_sampling_lost() {                  # 采样目录 <split>[-<标记>]-manifest.json：这份清单里槽数比读得回的记录多几个（打印一个数）
  python - "$1/$2" <<'PY'
import json, sys
from pathlib import Path
try:
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    lost = manifest["slots"] - min(manifest["raw_records"], manifest["graded_records"])
except (OSError, ValueError, KeyError, TypeError):
    lost = 0
print(lost if isinstance(lost, int) and lost > 0 else 0)
PY
}

rft_training_complete() {              # 检查点目录：训练做完过返回 0
  python - "$1" <<'PY'
import json, sys
from pathlib import Path
try:
    manifest = json.load(open(Path(sys.argv[1]) / "training-manifest.json"))
    complete = isinstance(manifest, dict) and manifest.get("status") == "complete"
except (OSError, ValueError):
    complete = False
sys.exit(0 if complete else 1)
PY
}

# 训练一组并在 dev 上选型。参数依次是：名称、数据目录、学习率、LoRA 秩、LoRA alpha、步序号；
# 多出来的参数原样传给训练脚本。已经选过型就只打印结论；没训完的目录挪到一边重训。
# 出错时和 run_rft.sh 原来的 arm() 一样直接 exit 1：训练或选型出错了，这一组就没有可下结论的东西
rft_arm() {
  local name="$1" data="$2" lr="$3" rank="$4" alpha="$5" step="$6"; shift 6
  local ckpt="$RUN/checkpoints/$name" select="$RUN/results/$name" total="${TOTAL:-?}"
  if [[ -s "$select/selection.json" ]]; then
    echo "===== $step/$total $name：已经选过型，跳过 ====="
  else
    echo "===== $step/$total $name：训练（lr $lr、秩 $rank）并在 dev 上选型 ====="
    if ! rft_training_complete "$ckpt"; then
      if [[ -d "$ckpt" ]]; then
        local moved="$ckpt.incomplete-$(date +%H%M%S)"
        mv "$ckpt" "$moved" || { echo "挪不开没训完的目录 $ckpt，把上面的报错贴给我"; exit 1; }
        echo "上次的训练没有完成，目录移到 $moved，重新训练"
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
  [[ -n "$BASE_REPORT" ]] || BASE_REPORT="$select/epoch-000-dev.json"     # 后面的组复用这一组评出的基座报告
  python scripts/tier_report.py "$select" || exit 1
  python - "$select/selection.json" "$name" <<'PY' \
    || { echo "读不出 $select/selection.json，把上面的报错贴给我"; exit 1; }
import json, sys
selection = json.load(open(sys.argv[1]))
chosen = next(row for row in selection["candidates"] if row["epoch"] == selection["selected_epoch"])
what = "保留基座" if selection["selected_epoch"] == 0 else f"选中 epoch {selection['selected_epoch']}"
print(f"结果：{sys.argv[2]} dev {selection['base_pass_at_1']:.3f} → {chosen['pass_at_1']:.3f}"
      f"（{what}；变好 {chosen['improved_tasks']} 道、变差 {chosen['regressed_tasks']} 道）")
PY
}

rft_eval_done() {                      # 输出目录 标签 dev|probe：这一部分评完了返回 0
  # 光有 summary 不算：它说的报告也得都在，写到一半断掉时留下的不能当成评过
  python - "$1" "$2" "$3" <<'PY'
import json, sys
from pathlib import Path
out, label, stem = sys.argv[1:4]
try:
    summary = json.loads((Path(out) / f"{label}-{stem}-summary.json").read_text(encoding="utf-8"))
except (OSError, ValueError):
    sys.exit(1)
if not isinstance(summary, dict):
    sys.exit(1)
names = [f"{label}-{stem}-greedy.json"]
if isinstance(summary.get("sampled"), dict):
    k = summary["sampled"].get("samples_per_task")
    if k:
        names.append(f"{label}-{stem}-sampled-k{k}.json")
sys.exit(0 if all((Path(out) / name).is_file() for name in names) else 1)
PY
}

rft_eval_unconfirmed() {                # 输出目录 标签：自检没能和基座比较时打印原因、返回 0；比较成了返回 1
  # eval_vllm.py 每次自检都写出 <标签>-probe-sanity.json。compared 不是 true（拿不到基座名、探针题里没有这一组
  # 训练过的题……）时它不作废、照常往下评；这里把原因交出去，调用方在汇总里注明这一组没能确认适配器是否生效
  python - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
try:
    sanity = json.loads((Path(sys.argv[1]) / f"{sys.argv[2]}-probe-sanity.json").read_text(encoding="utf-8"))
    if isinstance(sanity, dict) and sanity.get("compared") is not True:
        print(sanity.get("not_compared_reason") or "没能和基座比较")
        sys.exit(0)
except (OSError, ValueError):
    pass
sys.exit(1)
PY
}

rft_eval_dev() {                       # 模型名 标签 输出目录 [采样次数；0 或不给就是只贪心] [划分；不给就是 dev]
  # 划分是报告文件名里的一段（<标签>-<划分>-*.json），"评过没有"也按它查：dev 评过了不算 test 评过，反过来也一样
  local model="$1" label="$2" out="$3" samples="${4:-}" split="${5:-dev}"
  if rft_eval_done "$out" "$label" "$split"; then
    echo "$label 的 $split 评测已经有了，跳过"
    return 0
  fi
  local args=(--tasks-file "$TASKS" --vllm-url "http://127.0.0.1:${VLLM_PORT:-8000}" --model "$model"
              --label "$label" --output-dir "$out" --grade-workers "${GRADE_WORKERS:-4}" --split "$split")
  [[ -z "$samples" ]] || args+=(--samples "$samples")
  local how="贪心"
  [[ -z "$samples" || "$samples" == 0 ]] || how="贪心，每题再采样 $samples 次"
  echo "----- $label：$split 上评（$how）-----"
  mkdir -p "$out"
  python scripts/eval_vllm.py "${args[@]}"
}

rft_eval_probe() {                     # 模型名 标签 输出目录 探针题号文件 [自检用的训练数据]
  # 探针只做贪心。给了自检用的训练数据时 eval_vllm.py 先拿训练过的题把适配器和基座的输出比一遍，完全一样就以
  # 退出码 4 结束、不写报告和 summary，所以自检没过的组重跑时会重新自检（查的是服务现在的状态，代价只有十来次
  # 生成）。门槛就是 eval_vllm.py 的默认值 1：有一道题的输出和基座不同就算生效，探针题里训练过的题只有一两道时
  # 照样能过，不用按题数改门槛
  local model="$1" label="$2" out="$3" ids="$4" sanity="${5:-}"
  [[ -n "$ids" ]] || return 0
  if rft_eval_done "$out" "$label" probe; then
    echo "$label 的探针评测已经有了，跳过"
    return 0
  fi
  local args=(--tasks-file "$TASKS" --vllm-url "http://127.0.0.1:${VLLM_PORT:-8000}" --model "$model"
              --label "$label" --output-dir "$out" --grade-workers "${GRADE_WORKERS:-4}" --split train
              --ids-file "$ids" --tag probe --samples 0)
  [[ -z "$sanity" ]] || args+=(--sanity-tasks-file "$sanity")
  echo "----- $label：族内探针（贪心${sanity:+，先自检}）-----"
  mkdir -p "$out"
  python scripts/eval_vllm.py "${args[@]}"
}

rft_eval_vllm() {                      # 模型名 标签 输出目录 [采样次数] [探针题号文件] [自检用的训练数据] [划分；不给就是 dev]
  # 先评 dev（贪心加采样；最后给了划分就是评那个划分），再评探针（只贪心）。两部分各自"评过了就跳过"；
  # 返回值取两者退出码里较大的。
  # 要先评探针、自检没过就不评 dev（rft_vllm_eval.sh 对每一组就是这样做的），分两次调：先 rft_eval_probe，
  # 退出码是 4 就停下，否则再 rft_eval_vllm
  local model="$1" label="$2" out="$3" samples="${4:-0}" ids="${5:-}" sanity="${6:-}" split="${7:-dev}" rc=0 code=0
  rft_eval_dev "$model" "$label" "$out" "$samples" "$split" || rc=$?
  if [[ -n "$ids" ]]; then
    rft_eval_probe "$model" "$label" "$out" "$ids" "$sanity" || code=$?
    (( code <= rc )) || rc=$code
  fi
  return "$rc"
}
