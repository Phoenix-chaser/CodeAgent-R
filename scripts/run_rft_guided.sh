#!/usr/bin/env bash
# 阶梯上的 RFT 第 2 轮（引导采样）。第 1 轮里有一批题，基座采 8 次一次都没做对，验证组里就没有它们的正确答案。
# 这一轮带着提示对它们再采一遍：提示是参考解的开头（前缀提示），或者教师写的通用技能（技能提示）。采出判分
# 通过的答案后训练三组，看"难题的正确答案"是不是有用的训练信号：
#   D-prefix   第 1 轮的验证组 + 前缀提示引导出的正确答案
#   D-skill    第 1 轮的验证组 + 技能提示引导出的正确答案（SKILL=0，或可用的技能太少时没有这一组）
#   F-control  第 1 轮的验证组 + 这些引导题各一份随机取的、判分没通过的答案（训练加 --allow-unverified）
# 三组的超参相同（lr 1e-4、LoRA 秩 32、1 个 epoch），全部只在 dev 上选型：D 组明显高于 F 组，才说明正确答案有用。
# 做完再经本机 vLLM 服务评一遍（辅助指标：dev 上的采样通过率，以及对上一轮零通过的题的族内探针），只作旁证。
#
# 用法（在仓库根目录；第 1 轮要先做完，它的产物在 data/rft-round1）：
#   nohup bash scripts/run_rft_guided.sh > rft-guided.log 2>&1 &
#   WATCH=30 bash scripts/show_progress.sh rft-guided.log "" run_rft_guided.sh
# 在网页终端里粘贴时用下面这一行启动，重复粘贴也不会起第二个（整行粘贴）。它在 $HOME 里上一把锁：锁由 flock 自己
# 握着，不传给脚本和它启动的 vLLM，所以脚本跑完或者被 kill（包括 kill -9）锁就放开——再粘贴一次就是"中断后重跑"；
# 脚本跑着的时候再粘贴，只打印一句话，不动正在跑的那份日志。关掉网页也不影响：它在单独的会话里，不随终端退出。
# 粘贴以后终端里多出来的那一行分得清几种情况（要等脚本停下才打的那两句，终端还开着才看得到）：
#   已启动：…                                   拿到了锁，脚本在后台跑起来了；
#   已经在运行，不重复启动                      锁被占着：真的有一个在跑，这一次什么都没启动；
#   脚本已经以退出码 N 结束，现在不在运行。…     那一个已经停了（出错退出、被 kill，或者它自己以 1 结束，也就是
#                                               "没有全部完成"），不是还在跑：这一轮怎么样看跟在后面的日志最后
#                                               一行和进度面板，重新运行同一条命令只补没做完的；
#   已经跑完。…                                 脚本以 0 结束，后面同样跟着日志的最后一行。
# 脚本自己的退出码 1 和"锁被占"（flock -n 的退出码）都是 1，分不开：里面一层先把脚本的退出码加上 100 再交出来；
# 加锁本身没成功时 flock 的报错在上面，下面一句是"没有启动：加锁没有成功"。换了新的仓库包也是同一把锁：
#   setsid bash -c 'flock -n -o "$HOME/rft-guided.lock" bash -c "echo 已启动：在后台运行，日志在 rft-guided.log; bash scripts/run_rft_guided.sh > rft-guided.log 2>&1 < /dev/null; rc=\$?; [ \$rc = 0 ] || exit \$((rc + 100))"; status=$?; case $status in 0) echo "已经跑完。rft-guided.log 的最后一行：$(tail -n 1 rft-guided.log 2>/dev/null)";; 1) echo "已经在运行，不重复启动";; [2-9]|[1-9][0-9]) echo "没有启动：加锁没有成功，看上面 flock 的报错";; *) echo "脚本已经以退出码 $((status - 100)) 结束，现在不在运行。rft-guided.log 的最后一行：$(tail -n 1 rft-guided.log 2>/dev/null)";; esac' &
#
# 环境变量（都有默认值）：
#   MODEL / MODEL_REVISION                 基座模型和它的固定版本（不给就取仓库里的默认值）
#   TASKS_FILE=data/frozen-ladder.jsonl    冻结的任务文件
#   RUN_DIR=data/rft-guided                这一轮的运行目录
#   ROUND1_DIR=data/rft-round1             第 1 轮的运行目录（run_rft.sh 的 RUN_DIR）
#   ROUND1_PACKAGE=CodeAgent-R-v19          只用在提示里：缺少第 1 轮的产物时，提示从哪个包用 ln -s 把 data 接过来
#   RFT_TIERS=hard,variant                 引导哪些层的题（逗号分隔）
#   HINT_FRACTIONS=0.25,0.5,0.75           前缀提示各给参考解的多大比例
#   RFT_SAMPLES=4                          前缀引导：每种提示各采几次
#   SKILL=1                                1：教师写技能并做技能引导；0：不做，也没有 D-skill 组
#   SKILL_SAMPLES=8                        技能引导：每题采几次
#   MIN_SKILL_TASKS=5                      可用的技能少于这么多道时，不做技能引导（也没有 D-skill 组）
#   SKILL_JUDGE=1                           1：写完技能让教师自判一次；0：不自判（教师请求少一半，技能只过静态检查）
#   TEACHER_CONCURRENCY=2                   写技能时同时写几道题，也就是同一时刻最多有几个教师请求在飞。限额按账户算，
#                                          服务器上并发 4 时成片 429，所以默认 2；调大以前先想清楚
#   TEACHER_MIN_INTERVAL=1                  两次教师请求之间至少隔几秒
#   TEACHER_MAX_TOKENS=4096  TEACHER_TEMPERATURE=0.3  TEACHER_MAX_ROUNDS=2  TEACHER_TIMEOUT=120
#                                          教师接口的参数：同 TEACHER_MODEL、TEACHER_BASE_URL 一样是环境变量（write_skills.py
#                                          拿它们当默认值，也可以写在 .env 里）。教师先写一大段推理再给 JSON，上限 900 时
#                                          成片报"回复里没有 JSON 对象"，多半是没写完就到了上限，所以默认 4096；还被截断时
#                                          继续调大，原样重跑只重做出错的题。限速（429）和服务端错误时脚本自己长时间
#                                          重试（最长 30 分钟）：日志里一行一行写着"教师接口限速（429），等 N 秒后重试"，
#                                          不是卡死了。
#                                          TEACHER_EXTRA_JSON 可选：一个 JSON 对象，原样合并进每次请求（接口特有的参数）
#   VLLM_EVAL=1                            1：经本机 vLLM 服务评基座和各组（辅助指标）；0：不评
#   EVAL_SAMPLES=4                         vLLM 评测时 dev 上每题采样几次（0 表示只做贪心）
#   RFT_LR=1e-4  RFT_LORA_R=32  RFT_LORA_ALPHA=64  RFT_EPOCHS=1  GRAD_ACCUM=8  MAX_LENGTH=2048   三组共用的训练超参
#   VLLM_BIN=$HOME/vllm-qwen/bin/vllm      VLLM_PORT=8000
#   RFT_CONCURRENCY=24  RFT_GRADE_WORKERS=4  RFT_POLL_SECONDS=10    同 run_rft.sh
#   BASE_REPORT                            已有的基座 dev 报告；不给就用第 1 轮 A-verified 评出的那份，
#                                          连那份也没有时由第一组自己评；设成空（BASE_REPORT=）就是强制自己评
# SKILL=1 时教师的密钥从环境变量 TEACHER_API_KEY 读（还有 TEACHER_BASE_URL、TEACHER_MODEL）；当前目录有 .env 时
# 脚本先载入它，里面写的变量会盖过外面同名的环境变量。换了新的仓库包时 .env 不会跟过来（打包不带它）：从上一包
# 拷一份到当前目录，或者直接设环境变量。没有密钥（或者没装 openai SDK）时写技能做不成：上一跑留下的可用技能不够时，
# 这一轮照常做前缀引导和两组对照，但没有 D-skill 组，结束时不算是全部完成（见下）；把密钥配上再原样重跑，会把它补上。
# 每一步的产物已存在就跳过，中断后重新启动会接着跑。采样做没做完看清单里 槽数 = 采样数 = 判分数，不是只看清单在不在
# （旧版本的 sample_rft.py 读不回某些采样时也写清单）。采样或评测途中 vLLM 退出（显存不足、被系统杀掉）时，脚本
# 把它重新启动一次再接着做，每一步最多重启一次。和重跑有关的两条规矩：
#   技能引导采样一旦开始过，技能文件就不再改动（采样协议里记着它的摘要，变了就续不上）；没写成的题这一轮不做技能引导。
#   两次运行之间 SKILL 变了（上一跑没有密钥或技能太少；或者这一跑设了 SKILL=0）：F-control 按"现在有几个引导组"重新
#   配对、重新训练，原来的数据、检查点、选型结果和 vLLM 评测挪到名字后面加 .unpaired-时间戳。
# 结束时：要做的都做完了才是「===== 全部结束 =====」，退出码 0。有没做完的——技能引导没做成（没有密钥，或者教师
# 接口出错太多，可用的技能不够）、基座或各组的 vLLM 评测没评完——一样走到最后，但最后一行是
# 「===== 没有全部完成：… =====」，退出码 1（进度面板显示"已结束，但没有全部完成"）。vLLM 评测是辅助指标，
# 它没评完不影响上面 dev 上的选型结果。重新运行同一条命令只补没做完的。
cd "$(dirname "$0")/.."
# 教师接口的密钥等可以写在 .env 里。SKILL=1 时先载入它：set -a 让里面的变量也传给子进程。
# 放在 unset AGENTLAB_SANDBOX 前面：这台机器的内核不支持它，.env 里就算写了也不生效
if [[ "${SKILL:-1}" == 1 && -f .env ]]; then set -a; . ./.env; set +a; fi
unset AGENTLAB_SANDBOX
export PYTHONPATH=$PWD/src PYTHONUNBUFFERED=1 HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
state() { python src/agentlab/experiments/run_state.py "$@"; }
source scripts/rft_lib.sh
MODEL="${MODEL:-$(state default-model)}"
REVISION="${MODEL_REVISION:-$(state revision "$MODEL")}"
TASKS="${TASKS_FILE:-data/frozen-ladder.jsonl}"
RUN="${RUN_DIR:-data/rft-guided}"
ROUND1="${ROUND1_DIR:-data/rft-round1}"
PREVIOUS="${ROUND1_PACKAGE:-CodeAgent-R-v19}"   # 换了新的仓库包时，上一包的目录名：它的 data 直接接过来用
TIERS="${RFT_TIERS:-hard,variant}"
HINT_FRACTIONS="${HINT_FRACTIONS:-0.25,0.5,0.75}"
SAMPLES="${RFT_SAMPLES:-4}"                     # 前缀引导：每种提示各采几次
SKILL="${SKILL:-1}"
SKILL_SAMPLES="${SKILL_SAMPLES:-8}"
MIN_SKILL_TASKS="${MIN_SKILL_TASKS:-5}"
SKILL_JUDGE="${SKILL_JUDGE:-1}"                          # 0：写完技能不自判，教师请求少一半
# 教师接口的节流和输出上限（.env 里写了就用 .env 里的）：限额按账户算，默认值取得保守——要慢，不要成片出错
TEACHER_CONCURRENCY="${TEACHER_CONCURRENCY:-2}"
TEACHER_MIN_INTERVAL="${TEACHER_MIN_INTERVAL:-1}"
TEACHER_MAX_TOKENS="${TEACHER_MAX_TOKENS:-4096}"
VLLM_EVAL="${VLLM_EVAL:-1}"
EVAL_SAMPLES="${EVAL_SAMPLES:-4}"
LR="${RFT_LR:-1e-4}" RANK="${RFT_LORA_R:-32}" ALPHA="${RFT_LORA_ALPHA:-64}"
EPOCHS="${RFT_EPOCHS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}" MAX_LENGTH="${MAX_LENGTH:-2048}"
VLLM_BIN="${VLLM_BIN:-$HOME/vllm-qwen/bin/vllm}"
VLLM_PORT="${VLLM_PORT:-8000}"
URL="http://127.0.0.1:$VLLM_PORT"
CONCURRENCY="${RFT_CONCURRENCY:-24}" GRADE_WORKERS="${RFT_GRADE_WORKERS:-4}"
STARTED=$SECONDS
for switch in SKILL VLLM_EVAL SKILL_JUDGE; do    # 这几个开关写错了（例如 yes）不能悄悄当成关；前两个还决定有哪几步
  [[ "${!switch}" == 0 || "${!switch}" == 1 ]] || { echo "$switch 只能是 0 或 1（现在是 ${!switch}）"; exit 1; }
done
if [[ -z "${BASE_REPORT+set}" ]]; then    # 没给这个变量：第 1 轮有基座报告就用它；设成空（BASE_REPORT=）就是不用
  BASE_REPORT=""
  report="$ROUND1/results/A-verified/epoch-000-dev.json"
  [[ ! -f "$report" ]] || BASE_REPORT="$report"
fi                                      # 基座和 dev 题都和第 1 轮一样：第 1 轮评出的基座报告不用重评
ROUND1_SAMPLING="$ROUND1/sampling" ROUND1_SFT="$ROUND1/sft/verified"
SAMPLING="$RUN/sampling" SFT="$RUN/sft"
UNMATCHED="$SFT/F-control/unmatched-tasks.txt"
# 技能引导多三步（写技能、技能引导采样、D-skill 组），vLLM 评测多一步
TOTAL=$(( 9 + 3 * SKILL + VLLM_EVAL ))
STEP=0
INCOMPLETE=()        # 没做完的事（一句话一件）：有就不是"全部结束"，见文件末尾
mark() { STEP=$((STEP + 1)); echo "===== $STEP/$TOTAL $* ====="; }
skip_step() { STEP=$((STEP + 1)); echo "===== $STEP/$TOTAL $1：已完成，跳过 ====="; }

# 自己启动的 vLLM 不留在显卡上：出错退出、Ctrl+C、被 kill 时也关掉它（第 7 步正常关掉以后就没有可关的了）
stop_own_vllm() { [[ -z "${RFT_VLLM_PID:-}" ]] || rft_stop_vllm; }
trap stop_own_vllm EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mark "检查前提"
[[ -s "$TASKS" ]] || { echo "缺少任务文件 $TASKS：先把冻结的阶梯划分拷到这个路径，或用 TASKS_FILE 指定"; exit 1; }
[[ -n "$MODEL" && -n "$REVISION" ]] \
  || { echo "查不到默认模型或固定版本（$MODEL / $REVISION），把这一行贴给我"; exit 1; }
[[ -x "$VLLM_BIN" ]] || { echo "找不到 vLLM：$VLLM_BIN（用 VLLM_BIN 指定它的路径）"; exit 1; }
missing=()
for path in sampling/train-manifest.json sampling/train-graded.jsonl sft/verified/train.jsonl sft/verified/dev.jsonl; do
  [[ -s "$ROUND1/$path" ]] || missing+=("$ROUND1/$path")
done
if (( ${#missing[@]} )); then
  echo "缺少第 1 轮的产物（这一轮是在它的基础上做的）："
  printf '  %s\n' "${missing[@]}"
  echo "第 1 轮还没做完就先运行 bash scripts/run_rft.sh。换了新的仓库包时不用重跑：在仓库根目录把上一包的 data 接过来，"
  echo "  ln -s ../$PREVIOUS/data data"
  exit 1
fi
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
if [[ "$SKILL" == 1 && -z "${TEACHER_API_KEY:-}" && ! -s "$SAMPLING/train-guided-skill-protocol.json" ]]; then
  # 技能引导采样已经开始过的运行目录不再调教师（技能文件冻结），缺密钥就与此无关了
  echo "提醒：环境里没有 TEACHER_API_KEY，这次写不了技能提示（write_skills.py 会以退出码 2 结束）："
  echo "      上一跑没有留下够用的技能时，这一轮就没有技能引导采样、也没有 D-skill 组，结束时不算是全部完成"
  echo "      （密钥可以写在当前目录的 .env 里；换了新的仓库包时从上一包拷过来。配上以后原样重跑会把 D-skill 组补上）"
fi
lost="$(rft_sampling_lost "$ROUND1_SAMPLING" train-manifest.json)"
if [[ "$lost" =~ ^[1-9][0-9]*$ ]]; then
  # 引导哪些题、探针清单都从上一轮的这份清单来：它少记了几份采样，那几道题就被当成"一次都没通过"
  echo "注意：上一轮的采样清单对不上——$lost 个槽位的采样或判分记录读不回来（旧版本的 sample_rft.py 会这样），"
  echo "      那几道题可能被记成「一次都没通过」。重新运行 bash scripts/run_rft.sh 会只补这些槽位，做完再运行本脚本"
fi
rft_check_gpu_free || exit 1

mark "写探针题号文件（上一轮零通过的题，之后经 vLLM 评适配器时拿它们做族内探针）"
python - "$ROUND1_SAMPLING/train-manifest.json" "$TIERS" "$RUN/probe-ids.txt" <<'PY' \
  || { echo "探针题号文件没写成，停止。把上面的说明贴给我"; exit 1; }
import json, sys
from pathlib import Path


def tier_of(task_id):     # 和 src/agentlab/benchmark/tiers.py 是同一份定义；这里只用标准库，不去导入它
    for prefix, tier in (("h3_", "hard"), ("h3v_", "variant"), ("s3v_", "synth_variant"), ("s3_", "synth_hard")):
        if task_id.startswith(prefix):
            return tier
    return "default"


manifest_path, tiers, output = sys.argv[1], set(sys.argv[2].split(",")), Path(sys.argv[3])
zero_pass = json.loads(Path(manifest_path).read_text(encoding="utf-8"))["zero_pass_tasks"]
probe = sorted(task for task in zero_pass if tier_of(task) in tiers)
if not probe:
    # 引导采样只做这些题（sample_rft.py 同样是"零通过的题 ∩ 这些层"），一道都没有时后面每一步都会报错，不如停在这里
    print(f"上一轮一次都没通过的 {len(set(zero_pass))} 道题里没有属于 {'、'.join(sorted(tiers))} 层的：没有可引导的题，"
          "这一轮做不了。RFT_TIERS 写的是哪些层？")
    sys.exit(1)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("".join(f"{task}\n" for task in probe), encoding="utf-8")
print(f"探针题：上一轮零通过的 train 题里属于 {'、'.join(sorted(tiers))} 层的共 {len(probe)} 道，写在 {output}")
PY

# 开不了就返回非 0（原因和日志的末尾 rft_start_vllm 自己会打印）；开起来就把 SERVER 置 1
start_vllm() {
  rft_start_vllm "$RUN/vllm-sampling.log" || return 1
  SERVER=1
}
# 这一步要不要开服务：前缀引导采样没做完，或者要评的基座还有哪一部分没评。都做完了就不开（重跑时省下加载的几分钟）。
# 技能引导采样要不要做得等第 4 步写完技能才知道（可用的技能可能太少）：它要用服务的时候在第 6 步现开
sampling_todo=0
rft_sampling_complete "$SAMPLING" train-guided-prefix-manifest.json || sampling_todo=1
base_todo=0
if [[ "$VLLM_EVAL" == 1 ]]; then
  { rft_eval_done "$RUN/vllm/base" base dev && rft_eval_done "$RUN/vllm/base" base probe; } || base_todo=1
fi
SERVER=0
if (( sampling_todo || base_todo )); then
  mark "启动 vLLM 并先评基座"
  start_vllm || exit 1
  if [[ "$VLLM_EVAL" == 1 ]]; then
    echo "----- 先评基座（dev 上贪心加采样，再评探针；已经评过的部分会跳过）-----"
    rc=0; rft_eval_vllm "$MODEL" base "$RUN/vllm/base" "$EVAL_SAMPLES" "$RUN/probe-ids.txt" || rc=$?
    # 基座的数字只用来给后面各组作对照，没评完也不耽误采样和训练；但这一轮结束时不算是全部完成，重新运行时会只补没完成的
    (( rc == 0 )) || echo "基座的 vLLM 评测没完成（退出码 $rc），不影响后面的采样和训练；这一轮结束时不算是全部完成，重新运行本脚本时会只补没完成的"
  fi
else
  skip_step "启动 vLLM"
  echo "（前缀引导采样和基座评测都已完成，不用开服务；技能引导采样要用时到那一步再开）"
fi

if [[ "$SKILL" == 1 ]]; then
  mark "教师给零通过的题写技能提示"
  if [[ -s "$SAMPLING/train-guided-skill-protocol.json" ]]; then
    # 技能引导采样已经开始过：采样协议里记着技能文件的摘要，文件一变后面的采样就续不上（sample_rft.py 会以 2 退出）。
    # 而写技能每次都会把上一跑出错的题重写、追加进去，所以从这一刻起不再调它；出错的那几道题这一轮不做技能引导
    echo "技能引导采样已经开始过：技能文件保持原样，这次不再调用教师（上次没写成的题这一轮不做技能引导）"
  else
    rc=0 teacher_error=""      # 写技能整个没做成的原因（空串是没出这种事）
    judge=(--judge); (( SKILL_JUDGE )) || judge=(--no-judge)
    # 并发、请求间隔、输出上限由上面的环境变量给（默认 2、1 秒、4096）：限额按账户算，输出全花在推理上也常见
    python scripts/write_skills.py --tasks-file "$TASKS" \
      --zero-pass-from "$ROUND1_SAMPLING/train-manifest.json" --split train --tiers "$TIERS" \
      --graded "$ROUND1_SAMPLING/train-graded.jsonl" --output "$RUN/skills.jsonl" \
      --concurrency "$TEACHER_CONCURRENCY" --min-interval "$TEACHER_MIN_INTERVAL" --max-tokens "$TEACHER_MAX_TOKENS" \
      "${judge[@]}" || rc=$?
    if (( rc == 3 )); then
      # 3 是有题出错：写技能自己会续跑（有定论的不再写，出错的原样重跑会重做）——没写完的题这次不做技能引导，别的照常
      echo "技能提示没全部写成，继续（没写成的题不做技能引导）"
    elif (( rc != 0 )); then
      # 约定的退出码只有 0（全部有定论）和 3（有 error）。2 是参数或环境不对（没有 TEACHER_API_KEY、没装 openai SDK、
      # 题文件读不了）：教师接口根本没连上；别的退出码是它自己崩了。都不是"有几道题没写成"。上一跑留下的可用技能
      # 够用就照常用；不够时这一轮没有技能引导采样、也没有 D-skill 组——那是因为出了错，结束时不算是全部完成
      if (( rc == 2 )); then
        teacher_error="写技能没有做成：教师接口的参数或环境不对（write_skills.py 退出码 2），没有 D-skill 组"
        echo "写技能没有做成（write_skills.py 以退出码 2 结束，是参数或环境不对，见上面的报错）。"
      else
        teacher_error="写技能没有做成：write_skills.py 以退出码 $rc 结束（不是约定的 0 或 3），没有 D-skill 组"
        echo "写技能没有做成（write_skills.py 以退出码 $rc 结束，不是约定的 0 或 3，见上面的报错）。"
      fi
      echo "上一跑留下的可用技能不够时，这一次就没有技能引导采样、也不训 D-skill 组；处理好了原样重跑会把它补上"
    fi
  fi
  # 数的是 sample_rft.py --skills-file 真正会用的那些。文件末尾截在某个汉字中间的坏字节换成替换字符（那一行下一刻
  # 就因解析不了被跳过），不能让整个片段崩掉；数不出来就停，不能把空串当成 0 条、悄悄关掉技能引导
  counted="$(python - "$RUN/skills.jsonl" <<'PY'
import json, sys

# 同一道题以最后一行为准，status 是 ok 且有正文才算可用；errored 是最后一行是 error 的题数
latest = {}
try:
    with open(sys.argv[1], encoding="utf-8", errors="replace") as handle:
        for line in handle.read().split("\n"):       # 一条记录一行，行尾只认 "\n"
            try:
                record = json.loads(line)
            except ValueError:
                continue          # 上次写到一半的坏行
            if isinstance(record, dict) and record.get("task_id"):
                latest[record["task_id"]] = record
except OSError:
    pass                          # 文件还没有：一条也用不上
print(sum(record.get("status") == "ok" and bool(record.get("text")) for record in latest.values()),
      sum(record.get("status") == "error" for record in latest.values()))
PY
)"
  if (( $? )) || [[ ! "$counted" =~ ^([0-9]+)\ ([0-9]+)$ ]]; then
    echo "数不出可用的技能（$RUN/skills.jsonl 读不了，或者格式不对）：把上面的报错贴给我"; exit 1
  fi
  usable="${BASH_REMATCH[1]}" errored="${BASH_REMATCH[2]}"
  if (( usable < MIN_SKILL_TASKS )); then
    echo "可用的技能只有 $usable 条，少于 MIN_SKILL_TASKS=$MIN_SKILL_TASKS：技能引导的题太少，"
    echo "不做技能引导采样，也不训 D-skill 组"
    SKILL=0
    TOTAL=$(( TOTAL - 2 ))        # 少了后面的两步（技能引导采样、D-skill 组）；写技能这一步已经做了
    if [[ -n "$teacher_error" ]]; then  # 不（只）是技能写得不行，是写技能整个没做成：D-skill 组因此没有做
      INCOMPLETE+=("$teacher_error")
    elif (( errored )); then             # 是教师接口出过错：D-skill 组因此没有做。这一轮都不算是全部完成
      INCOMPLETE+=("技能引导没有做成：可用的技能只有 $usable 条、少于 MIN_SKILL_TASKS=$MIN_SKILL_TASKS，其中 $errored 道出错，没有 D-skill 组")
    fi
  fi
fi

# 一种提示的引导采样：只采上一轮零通过的题里属于 TIERS 的题。sample_rft.py 的退出码 3 表示还有槽位没完成，
# 原样重跑只补这些，所以最多试 3 次；退出码 2 是参数或协议不对，重试没有用，立刻停。
# 做完了的判据是 rft_sampling_complete（清单在，而且槽数 = 采样数 = 判分数），不是只看清单在不在。
# 参数：标记（输出文件名里 <split>-<标记>-* 的那一段）、简称、这一步的说明、其余原样传给采样脚本
guided_sample() {
  local tag="$1" short="$2" what="$3"; shift 3
  local rc=0 attempt restarts=0
  if rft_sampling_complete "$SAMPLING" "train-$tag-manifest.json"; then
    skip_step "$short"
    return
  fi
  if [[ -s "$SAMPLING/train-$tag-manifest.json" ]]; then
    echo "上次留下的$short清单里槽数和记录数对不上（旧版本的 sample_rft.py 读不回某些采样时也写清单）：只补缺的槽位，清单重写"
  fi
  mark "$what"
  if (( ! SERVER )); then          # 第 3 步没有开服务（重跑时前缀采样已经做完）：这次采样要用，现在开
    echo "（为这次采样启动 vLLM）"
    start_vllm || exit 1
  fi
  for attempt in 1 2 3; do
    rc=0
    python scripts/sample_rft.py --tasks-file "$TASKS" --split train --tiers "$TIERS" \
      --vllm-url "$URL" --model "$MODEL" --concurrency "$CONCURRENCY" --grade-workers "$GRADE_WORKERS" \
      --output-dir "$SAMPLING" "$@" \
      --only-zero-pass "$ROUND1_SAMPLING/train-manifest.json" --tag "$tag" || rc=$?
    (( rc != 0 )) || break
    if (( rc == 2 )); then
      # 2 是参数或采样协议不对（两次运行之间改了开关、技能文件变了之类）：重试多少次都一样，服务其实没有问题
      echo "$short没有做成：sample_rft.py 以退出码 2 结束，是参数或采样协议不对，重试没有用。把上面的报错贴给我"
      exit 1
    fi
    if ! rft_vllm_running && (( attempt < 3 )); then
      # 没跑完的时候 vLLM 已经不在了（显存不足、被系统杀掉）：对着一个不在的服务重试没有用，重新启动一次再试
      if (( restarts )); then
        echo "$short没跑完，vLLM 又不在了（已经重新启动过一次），停止。vLLM 日志最后 15 行："
        tail -n 15 "$RUN/vllm-sampling.log" | cut -c1-200
        exit 1
      fi
      restarts=1
      echo "$short没跑完的时候 vLLM 已经不在了。vLLM 日志最后 15 行："
      tail -n 15 "$RUN/vllm-sampling.log" | cut -c1-200
      echo "重新启动 vLLM 后再试（只补没完成的槽位）"
      rft_stop_vllm              # 引擎的子进程可能还占着显存：先清掉，等它放出来再启动
      rft_wait_gpu_free || exit 1
      start_vllm || exit 1
      continue
    fi
    echo "$short第 $attempt 次没跑完，5 秒后重试（只补没完成的槽位）"; sleep 5
  done
  if (( rc != 0 )); then
    echo "$short三次都没完成，停止。vLLM 日志最后 15 行："
    tail -n 15 "$RUN/vllm-sampling.log" | cut -c1-200
    exit 1                      # 退出时上面的 trap 会关掉自己启动的 vLLM
  fi
}
guided_sample guided-prefix "前缀引导采样" \
  "前缀引导采样（参考解的开头按 $HINT_FRACTIONS 各给一部分，每种 $SAMPLES 次）" \
  --hint-kind prefix --hint-fractions "$HINT_FRACTIONS" --samples "$SAMPLES"
if [[ "$SKILL" == 1 ]]; then
  guided_sample guided-skill "技能引导采样" "技能引导采样（每题 $SKILL_SAMPLES 次）" \
    --hint-kind skill --samples "$SKILL_SAMPLES" --skills-file "$RUN/skills.jsonl"
fi

if (( SERVER )); then
  mark "关闭 vLLM，等显存释放"
  rft_stop_vllm
  rft_wait_gpu_free || exit 1
else
  skip_step "关闭 vLLM"
  echo "（这次没有启动 vLLM）"
fi

# 准备一个引导组：第 1 轮的验证组加上这种引导采出的正确答案，重放验证后每题留一条。
# 参数：组名、这种引导的候选文件、[传给 prepare_data.py 的额外参数…]。已经准备过的目录不动
prepare_guided() {
  local name="$1" candidates="$2"; shift 2
  if [[ -s "$SFT/$name/dataset-manifest.json" ]]; then
    echo "$name 的训练数据已经有了，跳过准备"
    return
  fi
  if [[ -d "$SFT/$name" ]]; then    # prepare_data.py 最后才写 manifest，也只往空目录里写：没有 manifest 的是写到一半的
    local moved="$SFT/$name.incomplete-$(date +%H%M%S)"
    mv "$SFT/$name" "$moved" || { echo "挪不开上次没写完的目录 $SFT/$name，把上面的报错贴给我"; exit 1; }
    echo "上次准备的 $name 没有写完，目录移到 $moved，重新准备"
  fi
  python scripts/prepare_data.py --tasks-file "$TASKS" \
    --input "$ROUND1_SFT/train.jsonl" "$ROUND1_SFT/dev.jsonl" "$candidates" \
    --output-dir "$SFT/$name" --max-per-task 1 "$@" \
    || { echo "准备 $name 的训练数据失败，把上面的报错贴给我"; exit 1; }
}
drop_applied() {                   # 数据目录 题号文件：这份训练集里已经没有题号文件里的题时返回 0
  python - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path

try:
    tasks = {json.loads(line)["task_id"] for line in (Path(sys.argv[1]) / "train.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()}
    dropped = {line.strip() for line in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines() if line.strip()}
except (OSError, ValueError, KeyError, TypeError):
    sys.exit(1)
sys.exit(1 if tasks & dropped else 0)
PY
}
build_control() {                  # 引导组的目录…：基础记录加上每道引导题一份没通过的采样
  if [[ -d "$SFT/F-control" && ! -s "$SFT/F-control/dataset-manifest.json" ]]; then
    local moved="$SFT/F-control.incomplete-$(date +%H%M%S)"
    mv "$SFT/F-control" "$moved" || { echo "挪不开上次没写完的目录 $SFT/F-control，把上面的报错贴给我"; exit 1; }
    echo "上次准备的 F-control 没有写完，目录移到 $moved，重新准备"
  fi
  python scripts/rft_control.py --graded "$ROUND1_SAMPLING/train-graded.jsonl" \
    --base-dir "$ROUND1_SFT" --guided-from "$@" --output-dir "$SFT/F-control" --seed 42 \
    || { echo "配对对照没做成，停止。引导采样没有解出第 1 轮验证组之外的新题时就会这样："
         echo "这一轮没有可比较的引导组，没有往下训练的必要。把上面的报错贴给我"
         exit 1; }
}
data_paired() {                     # 引导组的目录…：F-control 正是按这几组配的（每一组的数据也都在）时返回 0
  python - "$SFT/F-control/train.jsonl" "$ROUND1_SFT/train.jsonl" "$@" <<'PY'
import json, sys
from pathlib import Path


def tasks(path):
    return {json.loads(line)["task_id"] for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()}


try:
    control = tasks(sys.argv[1])
    expected = tasks(sys.argv[2])                 # 配对对照的题目：基础集的题 ∪ 各个引导组的引导题
    for directory in sys.argv[3:]:
        expected |= tasks(Path(directory) / "train.jsonl")
except (OSError, ValueError, KeyError, TypeError):
    sys.exit(1)                                   # 有哪一份数据还没有（比如上一跑没有 D-skill）：没有配对
sys.exit(0 if control == expected else 1)
PY
}
paired_before() {                   # 原来的 F-control 是按哪几个引导组配的（读它的 manifest；读不出来就说不知道）
  python - "$SFT/F-control/dataset-manifest.json" <<'PY'
import json, sys
from pathlib import Path

try:
    names = [Path(path).name for path in json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["guided_from"]]
except (OSError, ValueError, KeyError, TypeError):
    names = []
print(f"是按 {'、'.join(names)} 配的" if names else "是按哪几个引导组配的看不出来（它的 manifest 里没记）")
PY
}
set_aside_control() {               # 说明：把和现在的引导组对不上的 F-control 挪开——数据、检查点、选型结果、vLLM 评测
  local stamp="$(date +%Y%m%d-%H%M%S)" item
  echo "F-control 和现在的引导组对不上（$1）。"
  echo "D 组和 F 组的题目要逐题对得上，所以原来的 F-control 挪开，重新配、重新训。"
  for item in "$SFT/F-control" "$RUN/checkpoints/F-control" "$RUN/results/F-control" "$RUN/vllm/F-control"; do
    [[ -e "$item" ]] || continue
    mv "$item" "$item.unpaired-$stamp" || { echo "挪不开 $item，把上面的报错贴给我"; exit 1; }
    echo "  $item 移到 $item.unpaired-$stamp"
  done
}
dataset_line() {                   # 数据目录 组名：一行结论，训练集有多少条、多少题，其中多少是引导题
  python - "$1" "$2" "$ROUND1_SFT/train.jsonl" <<'PY' || { echo "读不出 $1 里的训练集，把上面的报错贴给我"; exit 1; }
import collections, json, sys
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tier_of(task_id):     # 和 src/agentlab/benchmark/tiers.py 是同一份定义；这里只用标准库，不去导入它
    for prefix, tier in (("h3_", "hard"), ("h3v_", "variant"), ("s3v_", "synth_variant"), ("s3_", "synth_hard")):
        if task_id.startswith(prefix):
            return tier
    return "default"


TIERS, LABELS = ("default", "hard", "variant", "synth_hard", "synth_variant"), {
    "default": "原有题", "hard": "难题", "variant": "变体", "synth_hard": "合成难题", "synth_variant": "合成变体"}
root, name, base = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
train, dev = rows(root / "train.jsonl"), rows(root / "dev.jsonl")
base_tasks = {row["task_id"] for row in rows(base)}
guided = [row for row in train if row["task_id"] not in base_tasks]      # 引导题：第 1 轮的验证组里没有的题
tiers = collections.Counter(tier_of(row["task_id"]) for row in guided)
by_tier = "、".join(f"{LABELS[tier]} {tiers[tier]}" for tier in TIERS if tiers[tier])
print(f"结果：{name} 训练集 {len(train)} 条 / {len({row['task_id'] for row in train})} 题，"
      f"其中引导题 {len(guided)} 条{'（' + by_tier + '）' if by_tier else ''}；dev {len(dev)} 条")
PY
}

mark "准备训练数据（第 1 轮的验证组 + 引导出的正确答案，以及逐题配对的对照）"
# 引导组三样东西一一对应：组名（数据目录、检查点、选型结果都用它）、采样输出文件名里的标记、数据目录。
# SKILL=0 时只有前缀这一组
guided_groups=(D-prefix) guided_tags=(guided-prefix)
if [[ "$SKILL" == 1 ]]; then guided_groups+=(D-skill); guided_tags+=(guided-skill); fi
guided_dirs=()
for name in "${guided_groups[@]}"; do guided_dirs+=("$SFT/$name"); done
# 配对对照要拿引导组的训练集才知道哪些是引导题，所以总是后做。有的引导题在上一轮的采样里连一份"写完了、格式完整、
# 又没判分通过"的答案都没有（例如全是被截断的）时就配不上：这道题两组都不留（引导组带 --drop-tasks-file 重新准备），
# 否则两组的题目不一样。重跑时分四种情况：对照还没做过；做过、全都配上了、而且正是按现在这几个引导组配的；做过但
# 有配不上的（做到一半断了）；做过但不是按现在这几个引导组配的（两次运行之间 SKILL 变了：上一跑没有密钥或技能太少，
# 没做成 D-skill；或者这一跑设了 SKILL=0）。最后一种不能照用：那样 D 组和 F 组的题目又对不上
need_drop=""
if [[ -s "$SFT/F-control/dataset-manifest.json" ]]; then
  if [[ -s "$UNMATCHED" ]]; then
    need_drop="$UNMATCHED"          # 上次配过，有配不上的题，做到一半断了：接着往下做
  elif data_paired "${guided_dirs[@]}"; then
    for name in "${guided_groups[@]}"; do echo "$name 的训练数据已经有了，跳过准备"; done
    echo "F-control 的训练数据已经有了，跳过准备"
  else
    set_aside_control "这一次的引导组是 ${guided_groups[*]}；原来的 F-control $(paired_before)"
  fi
fi
if [[ -z "$need_drop" && ! -s "$SFT/F-control/dataset-manifest.json" ]]; then
  for index in "${!guided_groups[@]}"; do
    prepare_guided "${guided_groups[index]}" "$SAMPLING/train-${guided_tags[index]}-candidates.jsonl"
  done
  build_control "${guided_dirs[@]}"
  [[ ! -s "$UNMATCHED" ]] || need_drop="$UNMATCHED"
fi
pass=0
while [[ -n "$need_drop" ]]; do
  (( pass++ ))
  if (( pass == 1 )); then
    echo "有引导题配不上没通过的答案（题号在 $need_drop）：把它们从各个引导组里也去掉，两组才逐题对得上。"
  else
    echo "重新配对后又出现配不上的题（题号在 $need_drop；多半是两次运行之间多出来的引导组带进来的）：把它们也去掉，再配一次。"
  fi
  dropped="$(sort -u "$UNMATCHED")"        # 这一轮去掉的题；F-control 要删掉重建，先把清单记下来
  stamp="$(date +%Y%m%d-%H%M%S)"
  (( pass == 1 )) || stamp="$stamp-$pass"  # 同一组可能在两轮里各挪开一次：名字不能撞
  for index in "${!guided_groups[@]}"; do
    name="${guided_groups[index]}"
    if [[ -d "$SFT/$name" ]] && drop_applied "$SFT/$name" "$need_drop"; then continue; fi
    if [[ -d "$SFT/$name" ]]; then
      mv "$SFT/$name" "$SFT/$name.unmatched-$stamp" || { echo "挪不开 $SFT/$name，把上面的报错贴给我"; exit 1; }
      echo "  $name 的目录移到 $SFT/$name.unmatched-$stamp"
    fi
    prepare_guided "$name" "$SAMPLING/train-${guided_tags[index]}-candidates.jsonl" --drop-tasks-file "$need_drop"
  done
  rm -rf "$SFT/F-control"         # 它是按去掉之前的引导组配的：删掉，按新的引导组重新配
  build_control "${guided_dirs[@]}"
  [[ -s "$UNMATCHED" ]] || break
  # 上一跑留下的清单是按上一次的引导组配出来的：这次多出来的引导组会带进新的配不上的题。把它们并进清单再来一轮；
  # 配对只看每道题自己的采样，去过掉的题不会再配不上，所以每一轮至少多去掉一道题，不会有完没了
  added="$(comm -13 <(printf '%s\n' "$dropped" | sort -u) <(sort -u "$UNMATCHED"))"
  if [[ -z "$added" ]] || (( pass >= 3 )); then
    echo "去掉之后还是有配不上的题（题号在 $UNMATCHED）：两组的题目对不上，先停在这里。把上面的输出贴给我"
    exit 1
  fi
  echo "重新配对后新配不上的题：$(echo "$added" | tr '\n' ' ')"
  fresh="$(cat "$UNMATCHED")"   # 先读出来再往同一个文件里写：边读边截断会读到空的
  printf '%s\n%s\n' "$dropped" "$fresh" | sort -u > "$UNMATCHED"
done
for name in "${guided_groups[@]}" F-control; do
  dataset_line "$SFT/$name" "$name"
done

# 训练一组并在 dev 上选型（rft_arm：已经选过型就只打印结论）。三组的超参相同，步序号接着往下数
STEP=$((STEP + 1)); rft_arm D-prefix "$SFT/D-prefix" "$LR" "$RANK" "$ALPHA" "$STEP"
if [[ "$SKILL" == 1 ]]; then
  STEP=$((STEP + 1)); rft_arm D-skill "$SFT/D-skill" "$LR" "$RANK" "$ALPHA" "$STEP"
fi
STEP=$((STEP + 1)); rft_arm F-control "$SFT/F-control" "$LR" "$RANK" "$ALPHA" "$STEP" --allow-unverified

# 按题配对只保证到文件为止：训练入口还会丢掉超过 --max-length 的记录，同一道题的正确答案和没通过的答案长短不一，
# 可能一边留下、一边被丢。对照组的题目是各个 D 组引导题的并集，所以拿"各 D 组合起来"和它比：读各组训练清单里
# 实际加载的题核对一遍（rft_control.py 的说明里要求训练时核对条数），不一样就指出来
python - "$RUN/checkpoints" F-control "${guided_groups[@]}" <<'PY' \
  || echo "（没能核对各组训练时实际加载的题，把上面的报错贴给我）"
import json, sys
from pathlib import Path


def trained(root, name):      # 这一组训练时实际加载的题；没训过或清单里没记时是 None
    try:
        return set(json.loads((root / name / "training-manifest.json").read_text(encoding="utf-8"))["data"]["task_ids"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


root, control, groups = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
control_tasks = trained(root, control)
guided_tasks = {name: trained(root, name) for name in groups}
if control_tasks is not None and all(tasks is not None for tasks in guided_tasks.values()):
    together = set().union(*guided_tasks.values())
    names = "、".join(groups)
    if together == control_tasks:
        print(f"结果：训练时实际加载的题 {names} 和 {control} 逐题相同（{len(together)} 道）")
    else:
        print(f"结果：训练时实际加载的题 {names} 合起来和 {control} 不一样：只在 {names} 里的 "
              f"{sorted(together - control_tasks)}，只在 {control} 里的 {sorted(control_tasks - together)}。"
              f"多半是有记录超过 --max-length 被训练入口丢掉了，比较 D 组和 {control} 时要看住这几道题")
PY

mark "汇总主结果（都在 dev 上；test 这一轮不评）"
# 第 1 轮的两组在前（第一行是基座，后面各行都是相对它变的题数），后面是这一轮的各组
report_dirs=("$ROUND1/results/A-verified" "$ROUND1/results/C-control")
for name in "${guided_groups[@]}" F-control; do report_dirs+=("$RUN/results/$name"); done
report_args=()
for directory in "${report_dirs[@]}"; do
  if [[ ! -s "$directory/selection.json" ]]; then       # 还没选完的组（或 SKILL=0 时的 D-skill）不列：要说一声，表里没有它的行
    echo "$directory：还没有选型结果，不列进汇总"
    continue
  fi
  report_args+=("$directory")
done
if (( ${#report_args[@]} )); then
  python scripts/tier_report.py "${report_args[@]}" || echo "汇总没打印出来（把上面的报错贴给我）"
else
  echo "没有已经选完的组，汇总跳过。"
fi

if [[ "$VLLM_EVAL" == 1 ]]; then
  # 第 3 步没评完的基座也是没做完（那时只提醒了一句，没有停）。先查好：下面 rft_vllm_eval.sh 跑完以后紧接着就是
  # 最后一行，中间不再夹别的工作——进度面板是看日志里最后一个步骤标记来判断这一轮结束没有的
  base_missing=""
  rft_eval_done "$RUN/vllm/base" base dev || base_missing="dev"
  rft_eval_done "$RUN/vllm/base" base probe || base_missing="${base_missing:+$base_missing、}probe"
  [[ -z "$base_missing" ]] || INCOMPLETE+=("基座的 vLLM 评测没评完（$base_missing）")
  mark "经 vLLM 评各组选中的适配器（辅助指标；主结果是上面 HF 贪心那张表）"
  # D 组评探针前先自检适配器在 vLLM 里是否生效：训练过的题上适配器的输出和基座完全一样，这一组就作废。
  # 自检拿不到基座名时没法比较：不作废，照常评，rft_vllm_eval.sh 结束时说明没能确认
  sanity_groups="D-prefix"
  [[ "$SKILL" != 1 ]] || sanity_groups="D-prefix D-skill"
  rc=0
  # INCLUDE_BASE=0：基座第 3 步已经评过，外面环境里留着 INCLUDE_BASE=1 也不让这一步重评
  INCLUDE_BASE=0 EXPECT_REPRODUCE="$sanity_groups" PROBE_IDS_FILE="$RUN/probe-ids.txt" EVAL_SAMPLES="$EVAL_SAMPLES" \
    bash scripts/rft_vllm_eval.sh "$RUN" "${guided_groups[@]}" F-control || rc=$?
  if (( rc != 0 )); then
    # 辅助指标没评完不影响上面的主结果，但这一轮不能算"全部结束"：原因和该怎么做 rft_vllm_eval.sh 已经打印了
    INCOMPLETE+=("vLLM 评测没有全部完成（rft_vllm_eval.sh 以退出码 $rc 结束）")
  fi
fi
if (( ${#INCOMPLETE[@]} )); then
  echo "===== 没有全部完成：${INCOMPLETE[*]}；上面的 dev 选型结果不受影响，重新运行同一条命令只补没做完的 ====="
  exit 1                      # 退出时上面的 trap 会关掉自己启动的 vLLM
fi
echo "===== 全部结束 ====="
