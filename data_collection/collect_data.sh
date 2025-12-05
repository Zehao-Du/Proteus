#!/bin/bash
# collect_data.sh - 专门用于收集网络数据和制造混乱

# ==========================================
# 🔧 CONFIGURATION / 配置区域
# ==========================================

# 1. Traffic Generation Target / 流量生成目标地址
# 用于生成网络负载的大文件链接（建议使用大文件，如 ISO 镜像）。
TRAFFIC_URL="https://mirrors.ustc.edu.cn/ubuntu-releases/22.04/ubuntu-22.04.5-desktop-amd64.iso"

# 2. Data Output Path / 数据输出文件路径
# 指定生成的 CSV 文件名或路径 (默认为 net_data.csv)
DATA_OUTPUT_PATH="../data/net_data.csv"

# ==========================================
# END CONFIGURATION
# ==========================================

if [ "$EUID" -ne 0 ]; then
  echo "❌ Please run with sudo"
  # 提示用户如果想保留 conda 环境需加 -E
  echo "💡 Tip: Use 'sudo -E ./collect_data.sh' to preserve your current Python environment."
  exit 1
fi

# Ensure we are in the script's directory
cd "$(dirname "$0")" || exit
PROJECT_ROOT=$(pwd)
echo "📂 Working directory: $PROJECT_ROOT"

# ------------------------------------------
# 🐍 Python Detection (Auto-adapt)
# ------------------------------------------

# Automatically find python3 in the current PATH
PYTHON_EXEC=$(which python3)

# Check if python3 was found
if [ -z "$PYTHON_EXEC" ]; then
    echo "❌ Error: 'python3' not found in PATH."
    echo "   Please ensure python3 is installed or check your \$PATH."
    exit 1
fi

echo "🐍 Using Python: $PYTHON_EXEC"

# Add system packages to PYTHONPATH for BCC (eBPF tools usually live here)
# Even if using Conda, we often need the system BCC library.
export PYTHONPATH=$PYTHONPATH:/usr/lib/python3/dist-packages

# ------------------------------------------
# 🛑 Cleanup & Execution
# ------------------------------------------

# Cleanup function specifically for collection tools
cleanup() {
    echo ""
    echo "🛑 Stopping data collection..."
    pkill -f "ebpf_collector.py"
    pkill -f "chaos_maker.py"
    pkill -f "wget"
    echo "✅ Data collection stopped. Data saved to: $DATA_OUTPUT_PATH"
}
trap cleanup EXIT

echo "=================================================="
echo "   📡 TokenFlow - Data Collector"
echo "=================================================="

# 1. Clear old data
echo "🧹 Cleaning up old data..."
rm -f "$DATA_OUTPUT_PATH"

# 2. Start eBPF Agent
echo "📡 Starting eBPF Agent..."
echo "    -> Output file: $DATA_OUTPUT_PATH"

# ⚠️ 这里传入了 --csv 参数
$PYTHON_EXEC ebpf_collector.py --interval 0.5 --csv "$DATA_OUTPUT_PATH" > agent.log 2>&1 &
AGENT_PID=$!
echo "    -> Agent PID: $AGENT_PID"

# 3. Start Traffic & Chaos
echo "🌊 Starting Background Traffic & Chaos..."
echo "    -> Target: $TRAFFIC_URL"

# Download loop using the configured variable
(while true; do wget -q --timeout=5 --tries=2 -O /dev/null "$TRAFFIC_URL"; sleep 1; done) &

# Chaos maker
$PYTHON_EXEC chaos_maker.py > chaos.log 2>&1 &

echo "=================================================="
echo "✅ Collection is running!"
echo "📝 Logs: agent.log, chaos.log"
echo "📂 Output: $DATA_OUTPUT_PATH"
echo "⏳ Press Ctrl+C to stop collection when you have enough data."
echo "=================================================="

# Wait specifically for the agent. If agent dies, script exits.
wait $AGENT_PID