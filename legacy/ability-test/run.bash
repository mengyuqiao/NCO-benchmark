#!/bin/bash
# ===============================================
# 重复执行 run_models_fallback.py 十次（并行）
# 每次独立日志
# ===============================================
set -e
cd "$(dirname "$0")"

PYTHON="python3"
SCRIPT="run_deepseek.py"
REPEAT=10
mkdir -p logs

for ((i=1; i<=REPEAT; i++)); do
  log="logs/run_${i}.log"
  echo "▶️  Starting run ${i} (log: $log)"
  nohup $PYTHON -u "$SCRIPT" > "$log" 2>&1 &
  sleep 1   # 稍微错开启动时间
done

echo "🚀 All ${REPEAT} runs started in background."
echo "📂 Logs in $(pwd)/logs"
echo "💡 Use: tail -f logs/run_1.log  来看第1次的进度"
echo "💡 Use: ps aux | grep $SCRIPT  来查看运行中的进程"
echo "💡 Use: kill <PID>  来终止某个运行中的进程"
echo "==============================================="
echo "✅ Done."