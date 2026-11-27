#!/usr/bin/env bash
# 精度对照：用做完的那一轮 RFT 的训练数据，在另一种精度下把各组的适配器重新训一遍，再经本机 vLLM（bf16）评测。
# 为什么要做：那一轮 RFT 是在 NF4 下训练、NF4 下选型的；同一组适配器放到 bf16 的 vLLM 里，NF4 下的提升就没有了
# （基座在 bf16 下还比 NF4 下多通过 16 道）。这里回答的是：训练本身放到和部署一样的精度（或者更接近它的 int8）里，
# 提升还在不在。数据、超参都和原来一样，只换精度，所以结果能和原来那一轮逐项对照。
#
# 用法（在仓库根目录；先做完一轮 RFT）：
#   QUANT=bf16 nohup bash scripts/rft_precision_probe.sh > rft-precision-bf16.log 2>&1 &
#   WATCH=30 bash scripts/show_progress.sh rft-precision-bf16.log "" rft_precision_probe.sh
# bf16 放不下（训练时报显存不足，脚本会说明）时，换 QUANT=int8 再跑一次。在网页终端里粘贴时用下面这一行启动
# （整行粘贴；重复粘贴不会起第二个）：
#   cd ~/jupyterlab/CodeAgent-R-v24 && if pgrep -f 'rft_precision_probe[.]sh' > /dev/null; then echo "已经在运行，不重复启动"; else nohup bash scripts/rft_precision_probe.sh > rft-precision-bf16.log 2>&1 < /dev/null & echo "已启动"; fi
#
# 环境变量（都有默认值）：
#   QUANT=bf16                          训练时的精度：bf16（不量化，最接近部署）、int8（8 位，显存约一半）、nf4（和原来一样，用来对照）
#   SOURCE_DIR=data/rft-expanded-1127   做完的那一轮 RFT 的运行目录（取它 sft/verified、sft/control 下的训练数据）
#   ARMS="A-verified C-control"         重训哪些组；名字里带 control 的组用 sft/control 的数据，并加 --allow-unverified
#   RUN_DIR=data/rft-precision-$QUANT   这一跑的运行目录
#   TASKS_FILE=data/frozen-ladder-v2.jsonl   vLLM 评测用的任务文件，要和 SOURCE_DIR 那一轮用的是同一份
#   EPOCHS=1  LR=1e-4  LORA_R=32  LORA_ALPHA=64  GRAD_ACCUM=8  MAX_LENGTH=2048   同 RFT 那一轮（也可以沿用 RFT_LR 这类名字）
#   EVAL_SAMPLES=4                      vLLM 评测时 dev 上每题采样几次
#   VLLM_BIN=$HOME/vllm-qwen/bin/vllm  VLLM_PORT=8000
# 不做 NF4 下的 dev 选型（那一轮已经做过，而且慢）：只训 EPOCHS 个 epoch，直接拿最后一个 epoch 去评。
# 中断后原样重跑：训练完的组不重训，vLLM 评过的部分跳过。
cd "$(dirname "$0")/.."
unset AGENTLAB_SANDBOX
export PYTHONPATH=$PWD/src PYTHONUNBUFFERED=1 HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# 不量化时显存很紧：让分配器按需扩展段，减少碎片
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
source scripts/rft_lib.sh
state() { python src/agentlab/experiments/run_state.py "$@"; }

QUANT="${QUANT:-bf16}"
case "$QUANT" in
  bf16|int8|nf4) ;;
  *) echo "QUANT 只认 bf16、int8、nf4，现在是 $QUANT"; exit 1 ;;
esac
SOURCE_DIR="${SOURCE_DIR:-data/rft-expanded-1127}"
ARMS="${ARMS:-A-verified C-control}"
RUN="${RUN_DIR:-data/rft-precision-$QUANT}"
TASKS="${TASKS_FILE:-data/frozen-ladder-v2.jsonl}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-${RFT_LR:-1e-4}}"
LORA_R="${LORA_R:-${RFT_LORA_R:-32}}"
LORA_ALPHA="${LORA_ALPHA:-${RFT_LORA_ALPHA:-64}}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
EVAL_SAMPLES="${EVAL_SAMPLES:-4}"
export VLLM_BIN="${VLLM_BIN:-$HOME/vllm-qwen/bin/vllm}" VLLM_PORT="${VLLM_PORT:-8000}"
STARTED=$SECONDS
count() { echo $ARMS | wc -w; }
TOTAL=$(( $(count) + 3 ))
step=0
mark() { step=$((step + 1)); echo "===== $step/$TOTAL $1 ====="; }
elapsed() { echo "（已用 $(( (SECONDS - STARTED) / 60 )) 分钟）"; }

mark "检查前提"
[[ -n "$ARMS" ]] || { echo "ARMS 里一个组名都没有"; exit 1; }
for name in $ARMS; do
  [[ "$name" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "组名只能由字母、数字、- 和 _ 组成：$name"; exit 1; }
done
[[ -s "$TASKS" ]] || { echo "缺少任务文件 $TASKS：用 TASKS_FILE 指向那一轮 RFT 用的那份"; exit 1; }
[[ -x "$VLLM_BIN" ]] || { echo "找不到 vLLM：$VLLM_BIN（用 VLLM_BIN 指定它的路径）"; exit 1; }
rft_check_gpu_free || exit 1
echo "精度：$QUANT；数据来自 $SOURCE_DIR；重训的组：$ARMS；运行目录 $RUN"
mkdir -p "$RUN"
if [[ ! -s "$RUN/run-note.txt" ]]; then
  { echo "精度：$QUANT"; echo "数据来自：$(cd "$SOURCE_DIR" 2>/dev/null && pwd || echo "$SOURCE_DIR")"
    echo "任务文件：$TASKS"; echo "组：$ARMS"; } > "$RUN/run-note.txt"
fi
elapsed

data_dir_for() { case "$1" in *control*) echo "$SOURCE_DIR/sft/control" ;; *) echo "$SOURCE_DIR/sft/verified" ;; esac; }

for name in $ARMS; do
  mark "$name：在 $QUANT 下训练（lr $LR、秩 $LORA_R、$EPOCHS 个 epoch）"
  ckpt="$RUN/checkpoints/$name"
  data="$(data_dir_for "$name")"
  extra=()
  case "$name" in *control*) extra=(--allow-unverified) ;; esac      # 对照组的训练数据是没通过的答案
  if rft_training_complete "$ckpt"; then
    echo "训练已经完成过，跳过训练"
  else
    [[ -s "$data/train.jsonl" && -s "$data/dev.jsonl" ]] \
      || { echo "找不到 $data 下的 train.jsonl 和 dev.jsonl：用 SOURCE_DIR 指向做完的那一轮 RFT 的运行目录"; exit 1; }
    if [[ -d "$ckpt" ]]; then
      moved="$ckpt.incomplete-$(date +%H%M%S)"
      mv "$ckpt" "$moved" || { echo "挪不开没训完的目录 $ckpt，把上面的报错贴给我"; exit 1; }
      echo "上次的训练没有完成，目录移到 $moved，重新训练"
    fi
    train_args=(--data "$data/train.jsonl" --eval-data "$data/dev.jsonl"
      --model "$(state default-model)" --revision "$(state revision "$(state default-model)")"
      --output "$ckpt" --epochs "$EPOCHS" --lr "$LR" --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA"
      --batch-size 1 --grad-accum "$GRAD_ACCUM" --max-length "$MAX_LENGTH" --seed 42
      --quantization "$QUANT" "${extra[@]}")
    python scripts/train_sft.py "${train_args[@]}" --dry-run \
      || { echo "训练数据没通过检查，把上面的报错贴给我"; exit 1; }
    # 训练的输出同时存进日志；管道的退出码是 tee 的，训练自己的退出码要从 PIPESTATUS 里取
    python scripts/train_sft.py "${train_args[@]}" 2>&1 | tee "$RUN/train-$name.log"
    rc=${PIPESTATUS[0]}
    if (( rc != 0 )); then
      if grep -qiE 'out of memory|OutOfMemoryError|CUDA error' "$RUN/train-$name.log"; then
        echo "$name 的训练在 $QUANT 下显存不够。"
        if [[ "$QUANT" == bf16 ]]; then
          echo "bf16 的 9B 模型光权重就约 18 GB：换 QUANT=int8 再跑一次（脚本会提示那条命令），精度比 NF4 更接近 bf16。"
        fi
      fi
      echo "训练出错（退出码 $rc），日志在 $RUN/train-$name.log；把报错贴给我"
      exit "$rc"
    fi
    echo "结果：$name 在 $QUANT 下训练完成（$EPOCHS 个 epoch，数据 $data）；适配器在 $ckpt/epoch-$(printf '%03d' "$EPOCHS")/adapter"
  fi
  elapsed
done

mark "写各组的选型结果（直接取最后一个 epoch）"
python - "$RUN" "$QUANT" "$EPOCHS" $ARMS <<'PY' \
  || { echo "写选型结果时出错，把上面的报错贴给我"; exit 1; }
import json, sys
from pathlib import Path

run, quantization, epochs = sys.argv[1], sys.argv[2], int(sys.argv[3])
for name in sys.argv[4:]:
    out = Path(run) / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    path = out / "selection.json"
    if path.exists():
        print(f"{name}：选型结果已经有了，跳过")
        continue
    # rft_vllm_eval.sh 只读这几项：selected_adapter 是适配器的路径；configuration 里的 model 和 revision
    # 是训练时的基座，vLLM 要按同一个基座启动，适配器才对得上
    manifest = json.loads((Path(run) / "checkpoints" / name / "training-manifest.json").read_text(encoding="utf-8"))
    adapter = Path(run) / "checkpoints" / name / f"epoch-{epochs:03d}" / "adapter"
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"{name}：找不到 {adapter}")
    train_config = manifest.get("train_config") or {}
    path.write_text(json.dumps({
        "schema_version": 1, "selected_epoch": epochs, "selected_adapter": str(adapter),
        "selection_reason": "precision_probe: 直接取最后一个 epoch，没有在 NF4 下做 dev 选型",
        "configuration": {"model": manifest["model"],
                          "revision": manifest.get("resolved_revision") or manifest.get("revision"),
                          "quantization": quantization, "lr": train_config.get("lr"),
                          "lora_r": train_config.get("lora_r"), "epochs": epochs},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{name}：选型结果写好了，适配器 {adapter}")
PY
elapsed

mark "经 vLLM（bf16）评基座和各组选中的适配器"
# 基座也评一遍：这一份要和各组的适配器在同一套评测、同一个任务文件下比。评过的部分会跳过
INCLUDE_BASE=1 EVAL_SAMPLES="$EVAL_SAMPLES" TASKS_FILE="$TASKS" bash scripts/rft_vllm_eval.sh "$RUN" $ARMS
rc=$?
if (( rc != 0 )); then
  echo "vLLM 评测没有全部做完（退出码 $rc）：原因看上面的说明。处理以后原样重跑，训练不会重来，只补没评完的"
  exit "$rc"
fi
echo "结果：这一跑的 vLLM 报告在 $RUN/vllm；对照用 SOURCE_DIR 那一轮在 NF4 下的表"
echo "===== 全部结束 ====="
