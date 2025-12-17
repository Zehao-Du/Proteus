#!/bin/bash
# collect_data.sh - 自动收集网络拥塞控制训练数据 (30分钟版)

# ==========================================
# 🔧 CONFIGURATION / 配置区域
# ==========================================

# 1. Collection Duration / 收集时长
# 30分钟 = 1800秒
DURATION_MINUTES=30
DURATION_SECONDS=$((DURATION_MINUTES * 60))

# 2. Traffic Generation Target / 流量源
# 使用 Ubuntu ISO 镜像作为大文件下载源，稳定且速度快
TRAFFIC_URL="https://mirrors.ustc.edu.cn/ubuntu-releases/22.04/ubuntu-22.04.5-desktop-amd64.iso"

# 3. Data Output Path / 输出路径
DATA_OUTPUT_PATH="../data/train_data_congestion.csv"

# ==========================================
# END CONFIGURATION
# ==========================================

if [ "$EUID" -ne 0 ]; then
  echo "❌ Error: Please run with sudo"
  echo "💡 Tip: Use 'sudo -E ./collect_data.sh' to preserve python environment."
  exit 1
fi

cd "$(dirname "$0")" || exit
echo "📂 Working directory: $(pwd)"

# ------------------------------------------
# 🐍 Python Environment Check
# ------------------------------------------

PYTHON_EXEC=$(which python3)
if [ -z "$PYTHON_EXEC" ]; then
    echo "❌ Error: 'python3' not found."
    exit 1
fi

# 检查必要的 Python 脚本是否存在
if [ ! -f "ebpf_collector.py" ] || [ ! -f "realistic_congestion.py" ]; then
    echo "❌ Error: Missing python scripts!"
    echo "   Please ensure 'ebpf_collector.py' and 'realistic_congestion.py' are in this folder."
    exit 1
fi

# Add BCC path just in case
export PYTHONPATH=$PYTHONPATH:/usr/lib/python3/dist-packages

# ------------------------------------------
# 🛑 Cleanup Logic
# ------------------------------------------

cleanup() {
    echo ""
    echo "🛑 Stopping all processes..."
    # Kill the background jobs started by this script
    kill $(jobs -p) 2>/dev/null
    
    # Force kill specific names to be safe
    pkill -f "ebpf_collector.py"
    pkill -f "realistic_congestion.py"
    pkill -f "wget"
    
    # Reset network rules just in case
    # (Assuming chaos_maker has a cleanup, but running manually is safer)
    tc qdisc del dev eth0 root 2>/dev/null
    
    echo "✅ Collection finished."
    echo "📊 Data saved to: $DATA_OUTPUT_PATH"
    echo "📝 Logs: collector.log, congestion_maker.log"
}
trap cleanup EXIT INT TERM

echo "=================================================="
echo "   📡 Auto Data Collector (30 Mins)"
echo "=================================================="

# 1. Clear old data
rm -f "$DATA_OUTPUT_PATH" collector.log congestion_maker.log

# 2. Start Smart Agent (Monitor)
# 使用 --interval 1.0 以配合 Next_RTT 预测逻辑
echo "📡 Starting Smart Agent (eBPF)..."
$PYTHON_EXEC ebpf_collector.py --interval 0.05 --csv "$DATA_OUTPUT_PATH" > collector.log 2>&1 &
AGENT_PID=$!
echo "    -> PID: $AGENT_PID"

# Give agent a second to compile eBPF
sleep 3

# 3. Start Background Traffic (Load)
echo "🌊 Starting Traffic Generator (wget)..."
# Infinite loop downloading to /dev/null
(while true; do 
    wget -q --timeout=5 --tries=1 -O /dev/null "$TRAFFIC_URL"
    sleep 0.5
done) &

# 4. Start Chaos Maker (Simulation)
echo "😈 Starting Chaos Maker (Network Faults)..."
$PYTHON_EXEC realistic_congestion.py > congestion_maker.log 2>&1 &

echo "=================================================="
echo "✅ System Running! Timer started for $DURATION_MINUTES minutes."
echo "   Start Time: $(date +%H:%M:%S)"
echo "   End Time:   $(date -d "+$DURATION_MINUTES minutes" +%H:%M:%S)"
echo "=================================================="

# ------------------------------------------
# ⏳ Countdown Timer
# ------------------------------------------

REMAINING=$DURATION_SECONDS
while [ $REMAINING -gt 0 ]; do
    # Calculate progress
    MIN=$((REMAINING / 60))
    SEC=$((REMAINING % 60))
    
    # Update line in place
    printf "\r⏳ Time Remaining: %02d:%02d  (Rows collected: %s) " $MIN $SEC "$(wc -l < $DATA_OUTPUT_PATH 2>/dev/null || echo 0)"
    
    sleep 1
    REMAINING=$((REMAINING - 1))
    
    # Check if agent is still alive
    if ! kill -0 $AGENT_PID 2>/dev/null; then
        echo ""
        echo "❌ Critical Error: Smart Agent died unexpectedly! Check collector.log."
        exit 1
    fi
done

echo ""
echo "🎉 Time's up!"
# Trap will handle the cleanup now