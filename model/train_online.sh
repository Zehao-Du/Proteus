#!/bin/bash
# run_online_learning.sh - 自动处理 Conda 路径的启动脚本

# ==========================================
# 🔧 配置区域
# ==========================================
TRAFFIC_URL="https://mirrors.ustc.edu.cn/ubuntu-releases/22.04/ubuntu-22.04.5-desktop-amd64.iso"
PREDICTOR_SCRIPT="online_rtt_predictor.py"
CHAOS_SCRIPT="../data_collection/chaos_maker.py"
MODEL_FILE="best_lstm_grid_search.pth"
CHAOS_LOG="chaos_runtime.log"
# ==========================================

cd "$(dirname "$0")" || exit

# ----------------------------------------------------------
# 🧙‍♂️ 自动权限提升与 Python 路径探测 (核心修改)
# ----------------------------------------------------------

# 阶段 1: 如果当前是普通用户 (非 Root)
if [ "$EUID" -ne 0 ]; then
    echo "🔍 Checking Python environment..."
    
    # 1. 在普通用户环境下获取 Python 路径 (此时是 Conda Python)
    DETECTED_PYTHON=$(which python3)
    
    if [ -z "$DETECTED_PYTHON" ]; then
        echo "❌ Error: Could not find python3 in current environment."
        exit 1
    fi
    
    echo "✅ Detected Conda/User Python: $DETECTED_PYTHON"
    echo "🔒 Elevating privileges (sudo) to run eBPF..."
    
    # 2. 带着探测到的 Python 路径，自我重启进入 sudo 模式
    # 使用 'env' 命令将变量传递给 sudo 环境
    exec sudo env PYTHON_EXEC="$DETECTED_PYTHON" "$0" "$@"
    
    # 脚本到这里会停止，因为 exec 替换了当前进程
fi

# 阶段 2: 此时已经是 Root 用户了
# ----------------------------------------------------------

# 检查 PYTHON_EXEC 是否由阶段 1 传入
if [ -z "$PYTHON_EXEC" ]; then
    # 如果用户强行运行了 'sudo ./run.sh'，我们无法探测 Conda
    # 只能退回到系统 Python，或提示用户
    echo "⚠️  Warning: You ran 'sudo' manually. Conda path might be lost."
    echo "   Recommended: Run './run_online_learning.sh' WITHOUT sudo first."
    PYTHON_EXEC=$(which python3)
fi

echo "🐍 Using Python: $PYTHON_EXEC"

# 再次检查 Python 是否可用
if ! "$PYTHON_EXEC" --version > /dev/null 2>&1; then
    echo "❌ Error: The python executable '$PYTHON_EXEC' is not valid."
    exit 1
fi

# 检查脚本文件
if [ ! -f "$PREDICTOR_SCRIPT" ] || [ ! -f "$CHAOS_SCRIPT" ]; then
    echo "❌ Error: Scripts not found in current directory."
    exit 1
fi

# 检查模型
if [ ! -f "$MODEL_FILE" ]; then
    echo "⚠️  Warning: '$MODEL_FILE' not found. Starting from scratch."
else
    echo "📂 Found Model: $MODEL_FILE"
fi

# Add BCC path (为了兼容系统安装的 bcc)
export PYTHONPATH=$PYTHONPATH:/usr/lib/python3/dist-packages

# ------------------------------------------
# 🛑 清理函数
# ------------------------------------------
cleanup() {
    echo ""
    echo "🛑 Stopping Simulation..."
    kill $(jobs -p) 2>/dev/null
    pkill -f "$CHAOS_SCRIPT"
    pkill -f "wget"
    echo "🧹 Resetting network rules..."
    tc qdisc del dev eth0 root 2>/dev/null
    echo "✅ Done."
}
trap cleanup EXIT INT TERM

# ------------------------------------------
# 🚀 启动任务
# ------------------------------------------
echo "=================================================="
echo "   🧠 LSTM Online Learning (Conda Safe Mode)"
echo "=================================================="

echo "🌊 [1/3] Starting Traffic (wget)..."
(while true; do 
    wget -q --timeout=5 --tries=1 -O /dev/null "$TRAFFIC_URL"
    sleep 0.5
done) &

echo "😈 [2/3] Starting Chaos Maker..."
"$PYTHON_EXEC" "$CHAOS_SCRIPT" > "$CHAOS_LOG" 2>&1 &
echo "    -> Log: $CHAOS_LOG"

echo "🤖 [3/3] Starting Online Predictor..."
echo "--------------------------------------------------"
echo "   Running with: $PYTHON_EXEC"
echo "   (Ctrl+C to stop)"
echo "--------------------------------------------------"

# 这里使用变量里存储的 Conda Python 路径来运行脚本
"$PYTHON_EXEC" "$PREDICTOR_SCRIPT"