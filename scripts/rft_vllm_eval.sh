#!/usr/bin/env bash
# 给一轮 RFT 里各组选中的适配器做 vLLM 评测（辅助指标）：dev 上贪心加每题采样几次，再对"上一轮零通过的题"
# 做族内探针（贪心），最后打印汇总表。数字来自 vLLM（bf16 权重），只作旁证，主结果仍是 HF 上的贪心选型
# （那张表由 run_rft_guided.sh 第 12 步、run_rft.sh 第 7 步打印；单独看：python scripts/tier_report.py 选型目录…）。
# 用法：bash scripts/rft_vllm_eval.sh <运行目录> <组名>…
#       例如 bash scripts/rft_vllm_eval.sh data/rft-guided D-prefix D-skill F-control
# 可选的环境变量：
#   INCLUDE_BASE=1                        同时评基座（run_rft_guided.sh 第 3 步已经评过基座，那时不用再开）
#   PROBE_IDS_FILE=题号文件                评族内探针：上一轮零通过的题，一行一个 task_id
#   EXPECT_REPRODUCE="D-prefix D-skill"   这些组评探针前先自检：拿训练过的题比较适配器和基座的贪心输出，
#                                         完全一样（eval_vllm.py 退出码 4）这组的 vLLM 评测就作废。自检拿不到
#                                         基座名、找不到训练数据时没法比较：不作废，结束时说明没能确认
#   EVAL_SAMPLES=4                        dev 上每题采样几次（0 表示只做贪心）
# 其余同 run_rft.sh：MODEL、MODEL_REVISION、TASKS_FILE、VLLM_BIN、VLLM_PORT、RFT_GRADE_WORKERS、RFT_POLL_SECONDS。
# 评过的部分已在就跳过，中断后重新启动会接着评；一组出错不影响别的组。vLLM 中途退出（显存不足、被系统杀掉）时
# 重新启动一次：把被打断的那一组没评完的部分补上，再接着评剩下的组；每次运行最多重启一次，重启后 vLLM 又不在了
# 就不再重启，剩下的留给下一次运行。退出码：0 做完（个别组作废也算）；1 没评完或没法评（vLLM 起不来、显卡被占），
# 重新运行同一条命令会只补没完成的。看进度：
#   WATCH=30 bash scripts/show_progress.sh <日志> "" rft_vllm_eval.sh
cd "$(dirname "$0")/.."
unset AGENTLAB_SANDBOX
export PYTHONPATH=$PWD/src PYTHONUNBUFFERED=1 HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
source scripts/rft_lib.sh
state() { python src/agentlab/experiments/run_state.py "$@"; }
RUN="${1:?用法：bash scripts/rft_vllm_eval.sh <运行目录> <组名>…，例如 bash scripts/rft_vllm_eval.sh data/rft-guided D-prefix}"
shift
(( $# )) || { echo "用法：bash scripts/rft_vllm_eval.sh <运行目录> <组名>…（至少给一个组名）" >&2; exit 1; }
for name in "$@"; do    # 组名同时是 vLLM 里 LoRA 的名字和报告文件名的一段，eval_vllm.py 也只认这几种字符
  [[ "$name" =~ ^[A-Za-z0-9_-]+$ ]] \
    || { echo "组名只能由字母、数字、- 和 _ 组成：$name" >&2; exit 1; }
done
MODEL="${MODEL:-$(state default-model)}"
REVISION="${MODEL_REVISION:-$(state revision "$MODEL")}"
TASKS="${TASKS_FILE:-data/frozen-ladder.jsonl}"
VLLM_BIN="${VLLM_BIN:-$HOME/vllm-qwen/bin/vllm}"
VLLM_PORT="${VLLM_PORT:-8000}"
GRADE_WORKERS="${RFT_GRADE_WORKERS:-4}"
EVAL_SAMPLES="${EVAL_SAMPLES:-4}"
PROBE_IDS_FILE="${PROBE_IDS_FILE:-}"
INCLUDE_BASE="${INCLUDE_BASE:-0}"
EXPECT_REPRODUCE="${EXPECT_REPRODUCE:-}"
TOTAL=8
STARTED=$SECONDS
FAILED=0            # 有没评完的部分：不打印"全部结束"，以 1 退出
RESTARTED=0         # vLLM 中途退出后重新启动过没有：每次运行最多重启一次
GROUP_FAILED=0      # 当前这一组有没有没评完的部分
lora_args=()        # 启动 vLLM 时登记各组的 LoRA 用；中途退出重启时还要用它
VOID=()             # 评测作废的组和原因（只有一种：适配器的输出和基座完全一样）
UNCONFIRMED=()      # 没能确认适配器是否生效的组和原因（自检没法比较；不作废，照常评）
ADAPTER_GROUPS=()   # 有适配器、要评的组
declare -A ADAPTER= # 组名 → 适配器的绝对路径

# 自己启动的 vLLM 不留在显卡上：出错退出、Ctrl+C、被 kill 时也关掉它
stop_own_vllm() { [[ -z "${RFT_VLLM_PID:-}" ]] || rft_stop_vllm; }
trap stop_own_vllm EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

in_expect() {        # 组名是否列在 EXPECT_REPRODUCE 里
  local item
  for item in $EXPECT_REPRODUCE; do [[ "$item" == "$1" ]] && return 0; done
  return 1
}

eval_pending() {     # 标签 输出目录：还有没评完的部分（要开着 vLLM 做）返回 0
  local label="$1" out="$2"
  rft_eval_done "$out" "$label" dev || return 0
  [[ -n "$PROBE_IDS_FILE" ]] && ! rft_eval_done "$out" "$label" probe
}

echo "===== 1/$TOTAL 读取各组的选型结果 ====="
[[ -s "$TASKS" ]] || { echo "缺少任务文件 $TASKS：把冻结的阶梯划分放到这个路径，或用 TASKS_FILE 指定"; exit 1; }
if [[ -n "$PROBE_IDS_FILE" && ! -s "$PROBE_IDS_FILE" ]]; then
  echo "找不到探针题号文件 $PROBE_IDS_FILE：它由 run_rft_guided.sh 写在运行目录里的 probe-ids.txt"; exit 1
fi
# 每组取 selected_adapter（保留基座时是 null）。适配器是在哪个基座的哪个版本上训练的也一起带出来：
# vLLM 必须按那个基座启动，适配器才对得上。诊断信息走 stderr，stdout 只有最后一行
line="$(python - "$RUN" "$MODEL" "$REVISION" "$@" <<'PY'
import json, sys
from pathlib import Path

run, model, revision, names = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
rows, bases, found = [], {}, 0
for name in names:
    path = Path(run) / "results" / name / "selection.json"
    try:
        selection = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"{name}：读不出选型结果 {path}（{error}）", file=sys.stderr)
        rows.append((name, "-"))
        continue
    found += 1
    adapter = selection.get("selected_adapter") or "-"
    rows.append((name, adapter))
    recorded = selection.get("configuration") or {}
    if adapter != "-" and recorded.get("model") and recorded.get("revision"):
        bases.setdefault((recorded["model"], recorded["revision"]), []).append(name)
if found == 0:
    print(f"这些组一个都还没有选型结果：{' '.join(names)}（检查组名，或等 run_rft_guided.sh 把它们选完）",
          file=sys.stderr)
    sys.exit(1)
if len(bases) > 1:
    detail = "；".join(f"{m} @ {r}：{'、'.join(group)}" for (m, r), group in bases.items())
    print(f"各组适配器训练时的基座不一样，没法用同一个服务评：{detail}", file=sys.stderr)
    sys.exit(1)
serve_model, serve_revision = next(iter(bases), (model, revision))
# 字段之间用 ASCII 的单元分隔符（0x1f）：模型名、版本号、路径里都不会有它。不能用制表符：read 把连续的制表符
# 当成一个，空的版本号被吞掉，后面的字段整体前移一位（组名就被当成版本号交给了 vllm serve）
print("\x1f".join([serve_model, serve_revision] + [field for row in rows for field in row]))
PY
)" || { echo "读各组的选型结果时出错，把上面的报错贴给我"; exit 1; }
IFS=$'\x1f' read -r -a fields <<< "$line"
if (( ${#fields[@]} < 4 || ${#fields[@]} % 2 )); then    # 模型、版本，后面组名和适配器成对出现
  echo "读各组的选型结果时出错：第 1 步应当给出模型、版本和每组一对的组名、适配器，现在是 ${#fields[@]} 项，把上面的报错贴给我"
  exit 1
fi
serve_model="${fields[0]:-}" serve_revision="${fields[1]:-}"
for (( i = 2; i < ${#fields[@]}; i += 2 )); do
  name="${fields[i]}" adapter="${fields[i + 1]}"
  if [[ "$adapter" == "-" ]]; then
    echo "$name：dev 上保留了基座（没有适配器），不用评"
    continue
  fi
  [[ -z "${ADAPTER[$name]:-}" ]] || continue                     # 同一个组写了两遍：只登记一次
  [[ "$adapter" == /* ]] || adapter="$PWD/$adapter"       # vLLM 在哪个目录启动不一定，给它绝对路径
  if [[ ! -s "$adapter/adapter_config.json" ]]; then
    echo "$name：找不到适配器 $adapter（里面没有 adapter_config.json），这组不评"
    continue
  fi
  ADAPTER_GROUPS+=("$name"); ADAPTER["$name"]="$adapter"
  echo "$name：适配器 $adapter"
done
if [[ -n "$serve_model" && ( "$serve_model" != "$MODEL" || "$serve_revision" != "$REVISION" ) ]]; then
  echo "各组的适配器是在 $serve_model @ $serve_revision 上训练的，vLLM 按这个基座启动（环境里的是 $MODEL @ $REVISION）"
fi
[[ -z "$serve_model" ]] || MODEL="$serve_model"
[[ -z "$serve_revision" ]] || REVISION="$serve_revision"
[[ -n "$MODEL" && -n "$REVISION" ]] || { echo "查不到默认模型或固定版本（$MODEL / $REVISION），把这一行贴给我"; exit 1; }

echo "===== 2/$TOTAL 确定要评的对象 ====="
if (( ${#ADAPTER_GROUPS[@]} == 0 )) && [[ "$INCLUDE_BASE" != 1 ]]; then
  echo "各组都没有适配器，也没有要评基座：没有可做的 vLLM 评测。"
  echo "结果：没有适配器，vLLM 评测跳过"
  echo "===== 全部结束 ====="
  exit 0
fi
targets=()
[[ "$INCLUDE_BASE" != 1 ]] || targets+=(基座)
targets+=("${ADAPTER_GROUPS[@]}")
echo "要评的模型：${targets[*]}"
if [[ -n "$PROBE_IDS_FILE" ]]; then
  echo "族内探针：$PROBE_IDS_FILE；评探针前要自检的组：${EXPECT_REPRODUCE:-无}"
  if [[ "$INCLUDE_BASE" != 1 ]] && ! rft_eval_done "$RUN/vllm/base" base probe; then
    echo "（还没有基座的探针报告：汇总里各组的括号不写基座的通过数；要写的话加 INCLUDE_BASE=1）"
  fi
elif [[ -n "$EXPECT_REPRODUCE" ]]; then
  echo "EXPECT_REPRODUCE 里列了组，但没有给 PROBE_IDS_FILE：自检只在评探针时做，这次不会自检"
fi
needs_server=0
[[ "$INCLUDE_BASE" != 1 ]] || { eval_pending base "$RUN/vllm/base" && needs_server=1; }
for name in "${ADAPTER_GROUPS[@]}"; do
  eval_pending "$name" "$RUN/vllm/$name" && needs_server=1
done

if (( needs_server )); then
  echo "===== 3/$TOTAL 检查显卡 ====="
  rft_check_gpu_free || exit 1
  echo "===== 4/$TOTAL 启动 vLLM ====="
  if (( ${#ADAPTER_GROUPS[@]} )); then
    lora_args=(--enable-lora --max-lora-rank 64 --lora-modules)       # 请求里把 model 写成组名，就是评这一组的适配器
    for name in "${ADAPTER_GROUPS[@]}"; do lora_args+=("$name=${ADAPTER[$name]}"); done
  fi
  rft_start_vllm "$RUN/vllm-eval.log" "${lora_args[@]}" || exit 1
else
  echo "===== 3/$TOTAL 检查显卡：都评过了，不用启动 vLLM，跳过 ====="
  echo "===== 4/$TOTAL 启动 vLLM：跳过 ====="
fi

note_failure() {     # 名称 退出码
  echo "$1：vLLM 评测没完成（退出码 $2），重新运行同一条命令会只补没完成的"
  FAILED=1 GROUP_FAILED=1        # GROUP_FAILED：当前这一组有没有没评完的部分（一组评完时清零）
}
not_evaluated() {     # 名称：vLLM 不在了（而且已经重启过一次），这一部分没有评
  echo "$1：vLLM 不在了，没有评；重新运行同一条命令会只补没评完的"
  FAILED=1
}

revive_vllm() {      # 这次运行启动的 vLLM 已经退出时重新启动一次；服务在（本来就在，或者重启成功）返回 0
  (( needs_server )) || return 0
  rft_vllm_running && return 0
  echo "vLLM 已经不在了（多半是中途退出），日志最后 15 行："
  tail -n 15 "$RUN/vllm-eval.log" | cut -c1-200
  if (( RESTARTED )); then
    echo "vLLM 重新启动过一次，又不在了：不再重启，剩下的部分留给下一次运行（重新运行同一条命令会只补没评完的）"
    return 1
  fi
  RESTARTED=1
  echo "重新启动 vLLM，接着评剩下的部分"
  rft_stop_vllm                   # 引擎的子进程可能还占着显存：先清掉，等它放出来再启动
  rft_wait_gpu_free || return 1
  rft_start_vllm "$RUN/vllm-eval.log" "${lora_args[@]}"
}

if [[ "$INCLUDE_BASE" != 1 ]]; then
  echo "===== 5/$TOTAL 评基座：INCLUDE_BASE 不是 1，跳过 ====="
elif eval_pending base "$RUN/vllm/base"; then
  echo "===== 5/$TOTAL 评基座 ====="
  if revive_vllm; then
    rc=0; rft_eval_vllm "$MODEL" base "$RUN/vllm/base" "$EVAL_SAMPLES" "$PROBE_IDS_FILE" || rc=$?
    (( rc == 0 )) || note_failure "基座" "$rc"
  else
    not_evaluated "基座"
  fi
else
  echo "===== 5/$TOTAL 评基座：已经评过，跳过 ====="
fi

evaluate_group() {   # 组名 [再评一次]：评一组。自检没过的记进 VOID，自检没能比较的记进 UNCONFIRMED；没评完的部分由 note_failure 记下
  local name="$1" again="${2:-}" out="$RUN/vllm/$1" probe_todo=0 sanity="" rc=0 why
  probe_todo=0                                   # 探针还没评：先评它，自检没过就不往下评 dev
  [[ -z "$PROBE_IDS_FILE" ]] || rft_eval_done "$out" "$name" probe || probe_todo=1
  sanity=""
  if (( probe_todo )) && in_expect "$name"; then
    sanity="$RUN/sft/$name/train.jsonl"
    if [[ ! -s "$sanity" ]]; then
      # 没有训练数据就挑不出训练过的题，没法比较适配器和基座的输出。作废只有一种原因（输出和基座完全一样），
      # 所以这里不作废：照常评，结束时说明这一组没能确认适配器是否生效
      echo "$name：找不到自检用的训练数据 $sanity，没法比较适配器和基座的输出；不作废，照常评"
      UNCONFIRMED+=("$name（找不到自检用的训练数据）")
      sanity=""
    fi
  fi
  if [[ -n "$again" ]]; then    # 上一次评到一半 vLLM 退出了：评过的部分会跳过，只补没评完的
    echo "===== 6/$TOTAL $name：vLLM 中途退出，重新启动后再评一次 ====="
  else
    echo "===== 6/$TOTAL $name：$( (( probe_todo )) && echo "先探针后 dev" || echo "dev" ) ====="
  fi
  rc=0
  if (( probe_todo )); then
    rft_eval_probe "$name" "$name" "$out" "$PROBE_IDS_FILE" "$sanity" || rc=$?
    if (( rc == 4 )); then
      # 退出码 4 只有这一种原因：eval_vllm.py 已经把适配器和基座的输出比过了
      echo "$name：适配器在这几道题上的输出和基座完全一样，适配器在 vLLM 里没有生效，这组的 vLLM 评测作废"
      VOID+=("$name（适配器的输出和基座完全一样）")
      return
    fi
    (( rc == 0 )) || note_failure "$name 的探针" "$rc"
    if (( rc == 0 )) && [[ -n "$sanity" ]] && why="$(rft_eval_unconfirmed "$out" "$name")"; then
      # 自检没法和基座比较（拿不到基座名、探针题里没有这一组训练过的题……）：eval_vllm.py 没有作废，这里记下来
      echo "$name：没能确认适配器是否生效（$why），不作废，照常评"
      UNCONFIRMED+=("$name（$why）")
    fi
  fi
  rft_eval_vllm "$name" "$name" "$out" "$EVAL_SAMPLES" || rc=$?
  (( rc == 0 )) || note_failure "$name" "$rc"
}

for name in "$@"; do
  out="$RUN/vllm/$name"
  [[ -n "${ADAPTER[$name]:-}" ]] || continue        # 没有适配器的组在第 1 步已经说明
  if ! eval_pending "$name" "$out"; then
    echo "===== 6/$TOTAL $name：已经评过，跳过 ====="
    continue
  fi
  revive_vllm || { not_evaluated "$name"; continue; }   # 服务中途退出，而且已经重启过一次：留给下一次运行
  before=$FAILED GROUP_FAILED=0
  evaluate_group "$name"
  if (( GROUP_FAILED )) && ! rft_vllm_running; then
    # 这一组的评测没做完，vLLM 就不在了：多半是它中途退出（显存不足、被系统杀掉）。重新启动后把这一组
    # 没评完的部分再评一次，不必等到有人来重新运行
    echo "$name：没评完的时候 vLLM 已经不在了"
    if revive_vllm; then
      GROUP_FAILED=0
      evaluate_group "$name" "再评一次"
      (( GROUP_FAILED )) || FAILED=$before         # 补上了：这一组就不算没评完
    fi
  fi
done

if (( needs_server )); then
  echo "===== 7/$TOTAL 关闭 vLLM ====="
  rft_stop_vllm
else
  echo "===== 7/$TOTAL 关闭 vLLM：这次没有启动，跳过 ====="
fi

echo "===== 8/$TOTAL 汇总 ====="
summary_args=(--run "$RUN" --arms "$@")
[[ -z "$PROBE_IDS_FILE" ]] || summary_args+=(--probe-ids-file "$PROBE_IDS_FILE")
python scripts/vllm_summary.py "${summary_args[@]}" \
  || echo "汇总没打印出来（把上面的报错贴给我）；已经评完的报告都在 $RUN/vllm"

if (( FAILED )); then
  echo "结果：vLLM 评测没有全部完成，重新运行同一条命令会只补没完成的；已经评完的报告在 $RUN/vllm"
  exit 1
fi
ending="vLLM 评测做完了"
(( ${#VOID[@]} )) && ending+="；作废的组：${VOID[*]}"
(( ${#UNCONFIRMED[@]} )) && ending+="；没能确认适配器是否生效的组：${UNCONFIRMED[*]}"
echo "结果：$ending；报告在 $RUN/vllm"
echo "===== 全部结束 ====="
