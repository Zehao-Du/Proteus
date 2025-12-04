#!/bin/bash
if [ "$EUID" -ne 0 ]; then
  echo "❌ Please run with sudo"
  exit 1
fi

# Ensure we are in the script's directory
cd "$(dirname "$0")" || exit
PROJECT_ROOT=$(pwd)
echo "📂 Working directory: $PROJECT_ROOT"

# Detect Python
PYTHON_EXEC="/home/v-boxiuli/miniconda3/envs/smartnet/bin/python3"
if [ ! -f "$PYTHON_EXEC" ]; then
    PYTHON_EXEC=$(which python3)
fi

# Add system packages to PYTHONPATH for BCC
export PYTHONPATH=$PYTHONPATH:/usr/lib/python3/dist-packages

cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    pkill -f "smart_agent.py"
    pkill -f "hint_server.py"
    pkill -f "chaos_maker.py"
    pkill -f "wget"
    echo "✅ Done."
}
trap cleanup EXIT

echo "=================================================="
echo "   🚀 TokenFlow System (Integrated)"
echo "=================================================="

# 1. Start eBPF Agent
echo "📡 [1/5] Starting eBPF Agent..."
rm -f net_data.csv
$PYTHON_EXEC smart_agent.py --interval 0.5 > agent.log 2>&1 &
AGENT_PID=$!
echo "    -> Agent PID: $AGENT_PID"

# 2. Start Traffic & Chaos
echo "🌊 [2/5] Starting Traffic & Chaos..."
(while true; do wget -q --timeout=5 --tries=2 -O /dev/null "https://mirrors.ustc.edu.cn/ubuntu-releases/22.04/ubuntu-22.04.5-desktop-amd64.iso"; sleep 1; done) &
$PYTHON_EXEC chaos_maker.py > chaos.log 2>&1 &

# 3. Train Model
echo "🧠 [3/5] Waiting for data & Training..."
for i in {1..30}; do
    if [ -f "net_data.csv" ] && [ $(wc -l < net_data.csv) -gt 20 ]; then
        echo "    ✅ Data ready."
        break
    fi
    # Check if agent is actually running or crashed
    if ! pgrep -f "smart_agent.py" > /dev/null; then
        echo "❌ eBPF Agent died! Checking agent.log:"
        tail -n 10 agent.log
        exit 1
    fi
    echo "    ⏳ Waiting... ($i/30)"
    sleep 1
done
$PYTHON_EXEC train_model.py

# 4. Start Hint Server
echo "🔗 [4/5] Starting Hint Server..."
$PYTHON_EXEC hint_server.py > hint_server.log 2>&1 &
SERVER_PID=$!
echo "    -> Server PID: $SERVER_PID"

# 5. Start LLM Simulator
echo "🤖 [5/5] Starting LLM Simulator..."
sleep 2
$PYTHON_EXEC llm_simulator.py

wait $SERVER_PID
