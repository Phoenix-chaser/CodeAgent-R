#!/usr/bin/env bash
# 随时查看某个后台任务的进度。
# 用法：bash scripts/show_progress.sh [日志文件] [结果行的正则] [进程名的正则]
#       WATCH=30 bash scripts/show_progress.sh …    每 30 秒自动刷新，任务结束后自动退出
# 日志路径相对仓库根目录；后两个参数可省略。
# 状态看日志里最后一个步骤标记（"=====" 开头的行）：是「全部结束」才是"已完成"；是「没有全部完成：…」
# （run_rft.sh、run_rft_guided.sh 有没做完的事时这样结束，退出码 1）就显示"已结束，但没有全部完成"。
# 嵌在里面的脚本（例如第 13 步的 rft_vllm_eval.sh）自己也打「全部结束」，所以不能只看日志里有没有这一行。
cd "$(dirname "$0")/.."
LOG="${1:-thinking-pilot.log}"
PATTERN="${2:-(候选：|结果：|通过率|类别|^[ABC] =|^C 组|^难题|^变体|^base|^epoch [0-9]|环境报错|推理没写完|出错|没起来|失败|被占用|Error|Traceback)}"
PROC="${3:-}"
WATCH="${WATCH:-0}"
STOP=0

board() {
  clear
  printf '==================================================\n'
  printf '  进度：%s              %s\n' "$LOG" "$(date '+%H:%M:%S')"
  printf '==================================================\n'
  if [[ ! -s "$LOG" ]]; then echo "还没有日志：任务尚未启动"; return; fi
  last="$(grep -E '^=====' "$LOG" | tail -n 1)"
  if [[ "$last" == "===== 全部结束 =====" ]]; then
    echo "状态: 已完成"; STOP=1
  elif [[ "$last" == "===== 没有全部完成："*" =====" ]]; then
    echo "状态: 已结束，但没有全部完成（原因在最后一行；重新运行同一条命令只补没做完的）"; STOP=1
  elif [[ -n "$PROC" ]] && ! pgrep -af "$PROC" | grep -vq 'show_progress\.sh'; then
    # 本面板（以及它刚派生、还没换成 grep 的子进程）的命令行里也带着进程名，先把自己滤掉再判断
    echo "状态: 没有在运行（中途中断了，看最后几行找原因）"; STOP=1
  else
    echo "状态: 进行中"
  fi
  echo
  echo "步骤"
  mapfile -t steps < <(grep -E '^=====' "$LOG" | sed 's/^===== *//; s/ *=====$//')
  for ((i = 0; i < ${#steps[@]}; i++)); do
    if [[ "${steps[i]}" == "没有全部完成："* ]]; then mark="[没做完]"
    elif [[ "${steps[i]}" == "全部结束" ]] || (( i + 1 < ${#steps[@]} )); then mark="[完成]  "; else mark="[进行中]"; fi
    echo "  $mark ${steps[i]}"
  done
  echo
  echo "当前进度（最后一行是最新）"
  grep -E '\] [0-9]+/[0-9]+ \([0-9]+%\)' "$LOG" | tail -n 4 | cut -c1-120
  echo
  echo "已出的结果"
  grep -E "$PATTERN" "$LOG" | grep -v '^=====' | tail -n 14 | cut -c1-170
  echo
  echo "日志最后 3 行"
  grep -v 'Landlock 不可用\|falling back' "$LOG" | tail -n 3 | cut -c1-170
}

if [[ "$WATCH" == 0 ]]; then
  board
else
  # Ctrl+C 只是停止观看，不影响后台任务
  while true; do
    STOP=0
    board
    (( STOP )) && exit 0
    sleep "$WATCH"
  done
fi
