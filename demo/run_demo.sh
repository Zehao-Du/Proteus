#!/bin/bash
# run_demo.sh - 专门用于运行演示 (Server + Simulator)

# ==========================================
# 🔧 CONFIGURATION / 配置区域
# ==========================================

# 1. Model Paths / 模型文件路径
# 请确保这些文件已经通过 train.sh 生成
ISO_MODEL="../agent/isolation_forest.pkl"
GBDT_MODEL="../agent/gbdt_model.pkl"

# 2. Data Source / 数据源路径
# 必须与 collect_data.sh 中设置的路径一致
DATA_CSV="../data/net_data.csv"

# 3. Server Configuration / 服务器配置
SERVER_PORT=5000

# ==========================================
# END CONFIGURATION
# ==========================================

if [ "$EUID" -ne 0 ]; then
  echo "❌ Please run with sudo (Server needs permissions)"
  exit 1
fi

# Ensure we are in the script's directory
cd "$(dirname "$0")" || exit

# ------------------------------------------
# 🐍 Python Detection
# ------------------------------------------
PYTHON_EXEC=$(which python3)
if [ -z "$PYTHON_EXEC" ]; then
    echo "❌ Error: 'python3' not found."
    exit 1
fi

# ------------------------------------------
# 🛑 Cleanup Logic
# ------------------------------------------
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    pkill -f "hint_server.py"
    echo "✅ Done."
}
trap cleanup EXIT

echo "=================================================="
echo "   🚀 TokenFlow - Inference & Demo"
echo "=================================================="

# Optional: Check if model file exists
if [ ! -f "$ISO_MODEL" ] || [ ! -f "$GBDT_MODEL" ]; then
    echo "⚠️  Warning: Model files not found ($ISO_MODEL or $GBDT_MODEL)."
    echo "   Have you run './train.sh'?"
    echo "   (Server uses default fallback logic if models are missing)"
fi

if [ ! -f "$DATA_CSV" ]; then
    echo "⚠️  Warning: Data file '$DATA_CSV' not found."
    echo "   Make sure collection script is running or path is correct."
fi

# 1. Start Hint Server
echo "🔗 [1/2] Starting Hint Server..."
# ⚠️ Pass configured variables as arguments
$PYTHON_EXEC hint_server.py \
    --iso-model "$ISO_MODEL" \
    --gbdt-model "$GBDT_MODEL" \
    --data-path "$DATA_CSV" \
    --port "$SERVER_PORT" \
    > hint_server.log 2>&1 &

SERVER_PID=$!
echo "    -> Server PID: $SERVER_PID"

# Give server a moment to start
echo "⏳ Waiting for server to initialize..."
sleep 2

# 2. Start LLM Simulator
echo "🤖 [2/2] Starting LLM Simulator..."
# Assuming LLM simulator connects to localhost:5000 by default. 
# If llm_simulator.py also needs args, you can add them similarly.
$PYTHON_EXEC llm_simulator.py

# Keep script running if needed, or exit when LLM sim exits
wait $SERVER_PID