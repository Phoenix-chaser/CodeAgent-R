#!/usr/bin/env bash
# 思考模式实测：同一批 12 道难题 + 12 道单缺陷变体，按三种解码设置各评一次，用来决定评测协议。
#   A 现有协议：思考关闭 + 贪心
#   B 采样·不思考：temperature 0.7 / top_p 0.8 / top_k 20（模型卡对非思考模式的推荐）
#   C 采样·思考：  temperature 0.6 / top_p 0.95 / top_k 20（模型卡对思考模式的推荐），上限 PILOT_MAX
# 只用 train 和 dev 的题。每一组的结果文件存在且有效就跳过，中断后重新启动会接着跑。
cd "$(dirname "$0")/.."
unset AGENTLAB_SANDBOX
export PYTHONPATH=$PWD/src PYTHONUNBUFFERED=1 HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
state() { python src/agentlab/experiments/run_state.py "$@"; }
MODEL="$(state default-model)"
REVISION="$(state revision "$MODEL")"
RUN=data/thinking-pilot
PILOT_MAX="${PILOT_MAX:-8192}"
mkdir -p "$RUN/results"

echo "===== 1/5 选出 12 道难题和 12 道变体（只用 train 和 dev 的题） ====="
[[ -s data/frozen-ladder.jsonl ]] || { echo "缺少 data/frozen-ladder.jsonl，先按第 1 步把它拷进来"; exit 1; }
[[ -n "$MODEL" && -n "$REVISION" ]] || { echo "查不到默认模型或固定版本（$MODEL / $REVISION），把这一行贴给我"; exit 1; }
[[ -s "$RUN/pilot-tasks.jsonl" ]] || python - <<'PY'
import json
from pathlib import Path
run = Path("data/thinking-pilot")
frozen = [json.loads(line) for line in Path("data/frozen-ladder.jsonl").read_text(encoding="utf-8").splitlines()
         if line.strip()]
usable = [spec for spec in frozen if spec["split"] in ("train", "dev")]
parents = sorted((spec for spec in usable if spec["task_id"].startswith("h3_")),
                 key=lambda spec: spec["task_id"])
steps = sorted((spec for spec in usable if spec["task_id"].startswith("h3v_")),
               key=lambda spec: spec["task_id"])
# 变体取另外 12 道难题各自的第一道：不和选中的难题重复，难度落在难题和原有题之间
first_step = {}
for spec in steps:
    first_step.setdefault(spec["variant_of"], spec)
chosen = parents[:12] + [first_step[parent["task_id"]] for parent in parents[12:24]]
run.mkdir(parents=True, exist_ok=True)
with (run / "pilot-tasks.jsonl").open("w", encoding="utf-8") as handle:
    for spec in chosen:
        handle.write(json.dumps({**spec, "split": "ood"}, ensure_ascii=False, sort_keys=True) + "\n")
print(f"候选：难题 {len(parents)} 道、变体 {len(steps)} 道；各取前 12 道，共 {len(chosen)} 道")
PY

# 结果文件存在、且不是整份环境报错（比如显卡被别的进程占着）时，这一组才算做完
usable() {
  [[ -s "$1" ]] && python - "$1" <<'PY'
import json, sys
try:
    records = json.load(open(sys.argv[1]))["results"]
except (OSError, ValueError, KeyError):
    sys.exit(1)
sys.exit(0 if records and any(r.get("failure_type") != "infrastructure_error" for r in records) else 1)
PY
}

# 评一组：结果文件名、组别、说明、传给 evaluate.py 的参数
arm() {
  local name="$1" letter="$2" label="$3"; shift 3
  echo "===== $STEP 评 $letter 组（$name）：$label ====="
  if usable "$RUN/results/$name.json"; then echo "这一组已有结果，跳过"; return; fi
  python scripts/evaluate.py --tasks-file "$RUN/pilot-tasks.jsonl" \
    --split ood --model "$MODEL" --revision "$REVISION" --quantization nf4 --seed 42 \
    --output "$RUN/results/$name.json" "$@" \
    || { echo "（评测出错，把上面的报错贴给我）"; exit 1; }
  python - "$RUN/results/$name.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
results = data["results"]
def tier(task):
    return "难题" if task.startswith("h3_") else "变体"
broken = [r["task_id"] for r in results if r.get("failure_type") == "infrastructure_error"]
unfinished = sum(r.get("failure_type") == "thinking_unfinished" for r in results)
line = "结果："
line += "、".join(f"{name} {sum(r['strict_pass'] for r in results if tier(r['task_id']) == name)}"
                 f"/{sum(tier(r['task_id']) == name for r in results)}" for name in ("难题", "变体"))
line += f"；失败类型 {data['metrics']['failure_counts']}"
if data["generation"].get("thinking"):
    line += f"；推理没写完 {unfinished} 道"
if broken:
    line += f"；环境报错 {len(broken)} 道 {broken[:6]}（不是模型答错，把这些题贴给我）"
print(line)
PY
}

STEP=2/5
arm off A "现有协议，思考关闭 + 贪心，上限 2048" --temperature 0 --max-new-tokens 2048
STEP=3/5
arm off-sampled B "采样、不思考，temperature 0.7 / top_p 0.8 / top_k 20，上限 2048" \
  --temperature 0.7 --top-p 0.8 --top-k 20 --max-new-tokens 2048
STEP=4/5
arm on C "采样、思考，temperature 0.6 / top_p 0.95 / top_k 20，上限 $PILOT_MAX" \
  --thinking --temperature 0.6 --top-p 0.95 --top-k 20 --max-new-tokens "$PILOT_MAX"

echo "===== 5/5 汇总 ====="
python - <<'PY'
import json
run = "data/thinking-pilot/results"
arms = [("A", "off", "现有协议：思考关闭 + 贪心"),
        ("B", "off-sampled", "采样、不思考：temperature 0.7 / top_p 0.8 / top_k 20"),
        ("C", "on", "采样、思考：temperature 0.6 / top_p 0.95 / top_k 20")]
data = {key: json.load(open(f"{run}/{key}.json")) for _, key, _ in arms}
records = {key: {r["task_id"]: r for r in payload["results"]} for key, payload in data.items()}
def tier(task):
    return "难题" if task.startswith("h3_") else "变体"
def width(text):
    return sum(2 if ord(char) > 0x2E7F else 1 for char in text)
def cell(text, size):
    return text + " " * (size - width(text))
print("A =", arms[0][2])
print("B =", arms[1][2])
print(f"C = {arms[2][2]}，上限 {data['on']['generation']['max_new_tokens']} token")
print()
print(cell("类别", 8) + cell("A", 7) + cell("B", 7) + "C")
for name in ("难题", "变体"):
    row = [sum(r["strict_pass"] for r in records[key].values() if tier(r["task_id"]) == name)
           for _, key, _ in arms]
    print(cell(name, 8) + "".join(cell(f"{passed}/12", 7) for passed in row))
for letter, key, _ in arms:
    broken = sum(r.get("failure_type") == "infrastructure_error" for r in records[key].values())
    if broken:
        print(f"{letter} 组有 {broken} 道是环境报错，不算模型答错")
on = records["on"]
unfinished = [task for task, r in on.items() if r.get("failure_type") == "thinking_unfinished"]
tokens = sorted(r.get("output_tokens") or 0 for r in on.values())
print(f"C 组：输出 token 中位 {tokens[len(tokens) // 2]} / 最大 {tokens[-1]}；推理没写完 {len(unfinished)} 道 {unfinished}")
print("各组失败类型:", {letter: data[key]["metrics"]["failure_counts"] for letter, key, _ in arms})
print()
print("逐题（A / B / C；C 组括号里是输出 token 数）")
def show(record):
    if record["strict_pass"]:
        return "通过"
    text = record.get("failure_type") or record["verdict"]
    if record.get("failure_type") == "thinking_unfinished":
        text += f"({record.get('output_tokens')})"
    return text
for task in sorted(records["on"]):
    print("  " + cell(task, 30) + "".join(cell(show(records[key][task]), 28) for _, key, _ in arms))
PY
echo "===== 全部结束 ====="
