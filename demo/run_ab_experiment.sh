#!/bin/bash
# ============================================================
# A/B Experiment Runner Script
# 
# 这个脚本用于运行完整的 Pacing ON/OFF 对比实验
# 
# 使用方法:
#   bash run_ab_experiment.sh
#
# 前置条件:
#   1. vLLM 服务正在运行 (有网络感知调度)
#   2. eBPF Collector 正在运行 (sudo)
#   3. Hint Server 正在运行
# ============================================================

set -e

# ===================== 配置区 =====================
VLLM_URL="http://localhost:8000/v1"
HINT_URL="http://localhost:5000/hint"
SESSIONS_PER_GROUP=5
MAX_TOKENS=200
PROMPT="Write a detailed explanation of how machine learning models are trained. Include concepts like gradient descent, backpropagation, and optimization."

# 网络故障注入 (需要 sudo)
ENABLE_CHAOS=false
CHAOS_DELAY=100
CHAOS_LOSS=2.0
CHAOS_INTERFACE="eth0"
# ==================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "   eBPF-TokenFlow A/B Experiment Runner"
echo "=============================================="
echo ""

# 检查依赖服务
echo "🔍 检查依赖服务..."

# 检查 vLLM
if curl -s "$VLLM_URL/models" > /dev/null 2>&1; then
    MODEL=$(curl -s "$VLLM_URL/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else 'unknown')" 2>/dev/null || echo "unknown")
    echo "✅ vLLM 服务运行中 (模型: $MODEL)"
else
    echo "❌ vLLM 服务未运行！"
    echo "   请先启动 vLLM:"
    echo "   export VLLM_HINT_SERVER_URL=http://localhost:5000/hint"
    echo "   python -m vllm.entrypoints.openai.api_server --model <MODEL> --gpu-memory-utilization 0.4"
    exit 1
fi

# 检查 Hint Server (需要 A/B 版本)
if curl -s "$HINT_URL" > /dev/null 2>&1; then
    HEALTH=$(curl -s "$HINT_URL" | python3 -c "import sys,json; print(json.load(sys.stdin).get('health', 'N/A'))" 2>/dev/null || echo "N/A")
    MODE=$(curl -s "$HINT_URL" | python3 -c "import sys,json; print(json.load(sys.stdin).get('mode', 'unknown'))" 2>/dev/null || echo "unknown")
    echo "✅ Hint Server 运行中 (Health: $HEALTH, Mode: $MODE)"
    
    # 检查是否支持 A/B 模式切换
    if curl -s "http://localhost:5000/mode/status" > /dev/null 2>&1; then
        echo "✅ A/B 模式切换支持已启用"
    else
        echo "⚠️  Hint Server 不支持 A/B 模式切换"
        echo "   建议使用: python demo/hint_server_ab.py"
    fi
else
    echo "❌ Hint Server 未运行！"
    echo "   请先启动 A/B 版 Hint Server:"
    echo "   python demo/hint_server_ab.py &"
    exit 1
fi

echo ""
echo "📊 实验配置:"
echo "   - Sessions/组: $SESSIONS_PER_GROUP"
echo "   - Max Tokens: $MAX_TOKENS"
echo "   - Chaos 注入: $ENABLE_CHAOS"
echo ""

# 确认运行
read -p "🚀 开始实验? [Y/n] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    echo "已取消"
    exit 0
fi

# 创建输出目录
mkdir -p ab_results

# 运行实验
echo ""
echo "=============================================="
echo "   开始 A/B 实验"
echo "=============================================="

CHAOS_ARGS=""
if [ "$ENABLE_CHAOS" = true ]; then
    CHAOS_ARGS="--enable-chaos --chaos-delay $CHAOS_DELAY --chaos-loss $CHAOS_LOSS --chaos-interface $CHAOS_INTERFACE"
fi

python3 ab_experiment.py \
    --sessions "$SESSIONS_PER_GROUP" \
    --max-tokens "$MAX_TOKENS" \
    --vllm-url "$VLLM_URL" \
    --hint-url "$HINT_URL" \
    --prompt "$PROMPT" \
    --output-dir ab_results \
    $CHAOS_ARGS

echo ""
echo "=============================================="
echo "   🎉 实验完成！"
echo "=============================================="
echo ""
echo "📊 查看结果:"
echo "   streamlit run ab_dashboard.py"
echo ""

