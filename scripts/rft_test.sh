#!/usr/bin/env bash
# 对 RFT 某一轮里选定的那一组在 test 上评一次。test 只在定了配置之后评：先看完各组在 dev 上
# 的结果，再决定评哪一组，评完不要换一组再评。
# 用法：bash scripts/rft_test.sh <组名>        例如 bash scripts/rft_test.sh A-verified
#       RUN_DIR=… TASKS_FILE=… 可以改数据目录和冻结划分；已经评完的报告会保留，中断后接着评。
cd "$(dirname "$0")/.."
unset AGENTLAB_SANDBOX
export PYTHONPATH=$PWD/src PYTHONUNBUFFERED=1 HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
ARM="${1:?用法：bash scripts/rft_test.sh <组名>，例如 A-verified}"
RUN="${RUN_DIR:-data/rft-round1}"
TASKS="${TASKS_FILE:-data/frozen-ladder.jsonl}"
SELECTION="$RUN/results/$ARM/selection.json"
CKPT="$RUN/checkpoints/$ARM"
OUT="$RUN/results/$ARM/test"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' '; }

echo "===== 1/4 读取选型结果 ====="
[[ -s "$SELECTION" ]] || { echo "找不到 $SELECTION：这一组还没有选完型"; exit 1; }
[[ -s "$TASKS" ]] || { echo "缺少任务文件 $TASKS"; exit 1; }
read -r MODEL REVISION ADAPTER < <(python - "$SELECTION" "$CKPT" <<'PY'
import json, sys
selection = json.load(open(sys.argv[1]))
manifest = json.load(open(sys.argv[2] + "/training-manifest.json"))
print(manifest["model"], manifest.get("resolved_revision") or manifest["revision"],
      selection["selected_adapter"] or "-")
PY
)
if [[ "$ADAPTER" == "-" ]]; then
  echo "结果：$ARM 在 dev 上保留了基座（没有哪个 epoch 优于基座），test 不用评。"
  echo "===== 全部结束 ====="
  exit 0
fi
echo "模型 $MODEL @ $REVISION；选中的 adapter：$ADAPTER"

echo "===== 2/4 检查显卡 ====="
used="$(gpu_used)"
if [[ -n "${used:-}" ]] && (( used > 2000 )); then
  echo "显卡被占用（已用 ${used} MiB）。先停掉占显卡的程序再评："
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | head -n 5
  exit 1
fi

echo "===== 3/4 在 test 上评测基座和选中的模型（各约半小时）====="
eval_args=(--tasks-file "$TASKS" --split test --model "$MODEL" --revision "$REVISION"
  --training-data "$CKPT/training-manifest.json" --quantization nf4 --temperature 0
  --seed 42 --max-new-tokens "$MAX_NEW_TOKENS")
if [[ -s "$OUT/base.json" ]]; then
  echo "基座的 test 报告已经有了，跳过"
else
  python scripts/evaluate.py "${eval_args[@]}" --output "$OUT/base.json" \
    || { echo "基座评测出错，把上面的报错贴给我"; exit 1; }
fi
if [[ -s "$OUT/selected.json" ]]; then
  echo "选中模型的 test 报告已经有了，跳过"
else
  python scripts/evaluate.py "${eval_args[@]}" --adapter "$ADAPTER" --output "$OUT/selected.json" \
    || { echo "选中模型评测出错，把上面的报错贴给我"; exit 1; }
fi

echo "===== 4/4 比较 ====="
python scripts/compare.py --baseline "$OUT/base.json" --treatment "$OUT/selected.json" \
  --bootstrap-samples 10000 --seed 42 --output "$OUT/comparison.md" \
  || { echo "比较出错，把上面的报错贴给我"; exit 1; }
cat "$OUT/comparison.md"
python scripts/tier_report.py "base=$OUT/base.json" "selected=$OUT/selected.json" || exit 1
echo "结果：$ARM 的 test 报告在 $OUT（comparison.md、base.json、selected.json）"
echo "===== 全部结束 ====="
