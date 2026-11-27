#!/usr/bin/env bash
# 合成出题：用教师接口合成一批新题族，逐个审计后并入题池。只占 CPU 和教师接口，不占显卡，可以和显卡上的
# 实验同时跑。
# 题池只有 327 道题、难题只有 20 个来源族，RFT 看不出提升，根子在题太少。一批是 FAMILIES 个主题（五个出题方向
# 均分，每个方向配一个具体领域）。每个主题向教师要一个题族——一个程序：参考解、题面、公开测试、隐藏测试、若干
# mutant。回复解析后在沙箱里真判分；没被接受、但问题可以返修时让教师改一轮，还不行就拒绝。接受的题族 = 母题
# 1 道 + 每个可用的 mutant 一道单缺陷变体，最后装进阶梯文件（原有的题原样保留，新题加在后面）。
#
# 用法（在仓库根目录）：
#   bash scripts/run_synth_pool.sh                      第 1 批
#   BATCH=2 bash scripts/run_synth_pool.sh              下一批（上一次跑完时会提示下一批的 BATCH）
#   FAMILIES=3 BATCH=1 bash scripts/run_synth_pool.sh   只补第 1 批里 3 个还没有定论的题族，不要新主题
#                                                      （有几个这样的、该怎么写，run 结束时会提示）
# 教师的密钥写在当前目录的 .env 里（TEACHER_API_KEY=…），或者直接放在环境变量里。.env 不会跟到新的仓库包里
# （打包不带它）：换包以后拷一份过来。
# 教师接口的参数也靠环境变量（写在 .env 里就行；见下面环境变量里的 TEACHER_*）：TEACHER_TEMPERATURE=0.8、
# TEACHER_TIMEOUT=300（秒），TEACHER_MODEL、TEACHER_BASE_URL 同别的脚本。可选的 TEACHER_EXTRA_JSON 是一个 JSON 对象，
# 原样合并进每次请求：接口特有的参数不用改代码就能传。教师接口限速（429）或者出错（5xx）时脚本自己长时间重试
# （最长 30 分钟），日志里一行一行写着"教师接口限速（429），等 N 秒后重试"，不是卡死了。
# 看进度（日志名用你启动时重定向到的那个文件；docs/POSTTRAIN-GPU.md 里那条启动命令写的是 synth-batch1.log）：
#   bash scripts/show_progress.sh synth-batch1.log
#   WATCH=60 bash scripts/show_progress.sh synth-batch1.log '结果：|接受|拒绝' run_synth_pool.sh     每 60 秒刷新，跑完自动退出
# 怎么停：按一次 Ctrl-C，不再请求新的，正在做的几个题族做完就退；再按一次，或者 kill 这个脚本
# （pkill -f run_synth_pool.sh），立刻停，正在做的那几族的回答不保存。脚本被停掉时 python 跟着一起停，不会把它
# 成了孤儿、把这一批剩下的主题照样向教师要完。
# 在网页终端里粘贴时用下面这一行启动（整行粘贴；要指定批次就把 BATCH=2 写在最前面）。它在 $HOME 里上一把锁：
# 重复粘贴不会起第二个——锁被占着时只打一句"已经在运行"，不动正在跑的那份日志。脚本跑完或者被 kill 锁就放开，
# 再粘贴一次就是"重跑"：正常跑完的第 N 批不再向教师要题，只补后面没做完的。关掉网页也不影响：它在单独的会话里。
#   setsid bash -c 'flock -n -o "$HOME/synth-pool.lock" bash -c "echo 已启动：在后台运行，日志在 synth-pool.log; bash scripts/run_synth_pool.sh > synth-pool.log 2>&1 < /dev/null; rc=\$?; [ \$rc = 0 ] || exit \$((rc + 100))"; status=$?; case $status in 0) echo "已经跑完。synth-pool.log 的最后一行：$(tail -n 1 synth-pool.log 2>/dev/null)";; 1) echo "已经在运行，不重复启动";; [2-9]|[1-9][0-9]) echo "没有启动：加锁没有成功，看上面 flock 的报错";; *) echo "脚本已经以退出码 $((status - 100)) 结束，现在不在运行。synth-pool.log 的最后一行：$(tail -n 1 synth-pool.log 2>/dev/null)";; esac' &
#
# 环境变量（都有默认值）：
#   BATCH=1                                  第几批，从 1 起
#   FAMILIES=50                              这一批向教师要多少个题族
#   DIRECTIONS=all                           出题方向：all，或者逗号分隔的方向短名（state、tx、exc、lock、cond）
#   SYNTH_DIR=data/synth-pool                题族的记录（families/）、冻结的划分、新题文件都放在这里
#   PREV_PACKAGE=CodeAgent-R-v19              上一包在上一级目录里的名字。data 还不存在时，把 ../<它>/data 接过来当 data
#                                            （和引导那一轮共用：第 1 轮的产物和阶梯文件都在那里）
#   LADDER_FILE=data/frozen-ladder.jsonl     已有的题池文件（阶梯）；不存在就跳过第 5 步的导出（题族的记录都在，
#                                            把它放到这个路径以后原样重跑就导出，这一批不会再请求）
#   LADDER_OUT=data/frozen-ladder-v2.jsonl   阶梯加上新题以后写到哪。每批导出都把它更新成"阶梯 + 到现在为止全部
#                                            新题"；里面要是别的内容（不是这个脚本以前导出的）就不覆盖，退出码不变
#   TEACHER_CONCURRENCY=2                   同时处理几个题族，也就是同一时刻最多有几个教师请求在飞。限额按账户算，
#                                            调大它以前先想清楚（成片 429 时每等一次要 15 秒起步）
#   TEACHER_MIN_INTERVAL=1                  两次教师请求之间至少隔几秒
#   TEACHER_MAX_TOKENS=12000                每次回复的输出上限。教师先写一大段推理再给题族，回复被截断时调大它
#
# 重跑同一条命令：
#   已经有定论的题族（接受了，或者因为题族本身被拒绝）不再请求。
#   某一批用同样的参数正常跑完以后，原样重跑时这一批直接跳过（$SYNTH_DIR/batches.done 里记着一行），只补后面
#   没做完的。
#   教师调用出错、回复被截断或没法解析、判分没能完成的题族没有定论，不算用掉：只补它们就把 FAMILIES 写成它们的
#   个数（run 结束时会提示）；不管它们直接跑下一批也行。
#   退出码 3：一族都没接受，而且都不是题族本身的问题（教师接口出错、回复全都没法用……）。修好原因以后原样重跑，
#   就是把这一批的主题再要一次，不会要新主题。
#   show、status、export 出错只提示，不影响退出码。
cd "$(dirname "$0")/.."
# 当前目录有 .env 就载入：教师的密钥、地址写在里面，set -a 让它们也传给子进程。放在 unset AGENTLAB_SANDBOX
# 前面：这台机器的内核不支持它，.env 里就算写了也不生效
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
unset AGENTLAB_SANDBOX
export PYTHONPATH=$PWD/src PYTHONUNBUFFERED=1 HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

BATCH="${BATCH:-1}"
FAMILIES="${FAMILIES:-50}"
DIRECTIONS="${DIRECTIONS:-all}"
SYNTH_DIR="${SYNTH_DIR:-data/synth-pool}"
LADDER_FILE="${LADDER_FILE:-data/frozen-ladder.jsonl}"
LADDER_OUT="${LADDER_OUT:-data/frozen-ladder-v2.jsonl}"
# 教师接口的节流和输出上限（.env 里写了就用 .env 里的）：限额按账户算，默认值取得保守——要慢，不要成片出错
TEACHER_CONCURRENCY="${TEACHER_CONCURRENCY:-2}"
TEACHER_MIN_INTERVAL="${TEACHER_MIN_INTERVAL:-1}"
TEACHER_MAX_TOKENS="${TEACHER_MAX_TOKENS:-12000}"

# 脚本被停掉时把正在跑的 python 一起停掉。不这么做的话 python 成了孤儿，会把这一批剩下的主题照样向教师要完：
# 钱照算，这一批的"跑完"标记又写不上，之后原样重跑还要再来一批。
#   Ctrl-C：终端同时把信号发给了 python，第一次它自己会停掉排着的、等正在做的做完，这里只提醒一句；
#   再按一次，或者 kill 这个脚本（SIGTERM）、终端关闭（SIGHUP）：立刻把 python 停掉
RUN_PID=""
INTERRUPTED=0
stop_run() {
  local pid="$RUN_PID"
  RUN_PID=""
  [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null
  return 0
}
on_interrupt() {
  if (( INTERRUPTED )) || [[ -z "$RUN_PID" ]]; then stop_run; exit 130; fi
  INTERRUPTED=1
  echo "正在停：排着的主题不再请求，正在进行的题族做完就退出；再按一次 Ctrl-C 立刻停（它们这一次的回答不保存）。"
}
trap on_interrupt INT
trap 'stop_run; exit 143' TERM
trap 'stop_run; exit 129' HUP
trap stop_run EXIT

echo "===== 1/5 检查前提 ====="
# data 要在 mkdir 之前接好，不然 mkdir 会把它建成本包里的真实目录。已经是真实目录（上一回没接就跑了出题）、找不到
# 阶梯文件、而上一包的数据明明在：停在这里。再往下跑，引导那一轮的启动命令就建不了链接，会停在它的第 1 步
PREV_DATA="../${PREV_PACKAGE:-CodeAgent-R-v19}/data"
if [[ ! -e data && ! -L data && -d "$PREV_DATA" ]]; then
  ln -s "$PREV_DATA" data
  echo "已经把 data 接到 $PREV_DATA（和引导那一轮共用上一包的数据）"
elif [[ -e data && ! -L data && ! -e "$LADDER_FILE" && -d "$PREV_DATA" ]]; then
  echo "data 是本包里的真实目录，不是接到 $PREV_DATA 的链接，里面也没有阶梯文件 $LADDER_FILE。"
  echo "引导那一轮和这一轮都要用上一包的数据：先把它挪开（mv data data.local），再运行一次，脚本会自己把链接建好。"
  exit 1
fi
mkdir -p "$SYNTH_DIR"
if [[ -z "${TEACHER_API_KEY:-}" ]]; then
  echo "环境里没有 TEACHER_API_KEY：把 .env 放到项目目录（仓库根，和 scripts 并列），里面写 TEACHER_API_KEY=…；"
  echo "或者先 export TEACHER_API_KEY，再运行同一条命令。"
  exit 1
fi
echo "第 $BATCH 批：向教师要 $FAMILIES 个题族（方向 $DIRECTIONS），记录在 $SYNTH_DIR"

echo "===== 2/5 向教师要题族并逐个审计（每族最多 2 轮）====="
# 正常跑完的批次记在这个文件里，一行一个，批号和参数都写上：同一行已经在里面，这一批就不再请求
STAMP="batch=$BATCH families=$FAMILIES directions=$DIRECTIONS"
DONE="$SYNTH_DIR/batches.done"
if [[ -s "$DONE" ]] && grep -qxF "$STAMP" "$DONE"; then
  echo "这一批用同样的参数正常跑完过，跳过；再出一批用 BATCH=$((BATCH + 1))。"
  echo "要补做没有定论的题族，照 run 上次打印的提示把 FAMILIES 写成它们的个数再跑。"
else
  # python 放到后台再 wait：前台命令没做完以前 bash 不执行 trap，放在前台的话 kill 这个脚本要等 python 跑完
  # 才轮得到。SYNTH_POOL_STOP_WITH_PARENT 让 python 盯着父进程：脚本被 kill -9 时 trap 也执行不了，
  # 父进程一没它就自己停。输出同时存进这一批自己的日志。
  SYNTH_POOL_STOP_WITH_PARENT=1 python scripts/synth_pool.py run --output-dir "$SYNTH_DIR" --batch "$BATCH" \
    --families "$FAMILIES" --directions "$DIRECTIONS" \
    --concurrency "$TEACHER_CONCURRENCY" --min-interval "$TEACHER_MIN_INTERVAL" --max-tokens "$TEACHER_MAX_TOKENS" \
    > >(tee "$SYNTH_DIR/batch-$BATCH.log") 2>&1 &
  RUN_PID=$!
  wait "$RUN_PID"
  code=$?
  while (( code > 128 )) && kill -0 "$RUN_PID" 2>/dev/null; do    # wait 被 Ctrl-C 打断而 python 还在跑：接着等
    wait "$RUN_PID"
    code=$?
  done
  RUN_PID=""
  for _ in {1..40}; do                                            # 等 tee 把最后几行写完，步骤标记才不会插到它们前面
    pgrep -P "$$" -x tee >/dev/null || break
    sleep 0.05
  done
  if (( code != 0 )); then
    echo "第 $BATCH 批没有跑完（退出码 $code）：原因看上面的说明。修好以后原样重跑——已经有定论的题族不会再请求。"
    exit "$code"
  fi
  echo "$STAMP" >> "$DONE"
fi

echo "===== 3/5 汇总这一池 ====="
python scripts/synth_pool.py status --output-dir "$SYNTH_DIR" \
  || echo "统计没打印出来（不影响后面的步骤；把上面的报错贴给我）"

echo "===== 4/5 抽查 3 个已接受的题族 ====="
python scripts/synth_pool.py show --output-dir "$SYNTH_DIR" --sample 3 \
  || echo "抽查没打印出来（不影响导出；把上面的报错贴给我）"

echo "===== 5/5 并入阶梯 ====="
if [[ -s "$LADDER_FILE" ]]; then
  # 导出失败不改退出码：题族的记录都还在，照上一行的说明处理以后原样重跑接着导出（这一批不会再请求）
  python scripts/synth_pool.py export --output-dir "$SYNTH_DIR" --ladder "$LADDER_FILE" \
    --tasks-out "$SYNTH_DIR/synth-pool.jsonl" --ladder-out "$LADDER_OUT" \
    || echo "导出没有完成（原因见上一行）。处理以后原样重跑，第 $BATCH 批不会重新请求。"
else
  echo "阶梯文件 $LADDER_FILE 不存在：跳过导出。接受的题族都在 $SYNTH_DIR/families，"
  echo "要并入阶梯时把它拷到这个路径，再原样重跑（第 $BATCH 批不会重新请求）。"
fi
echo "===== 全部结束 ====="
