#!/bin/bash
# run_real_llm.sh - 运行真实 LLM 客户端（vLLM 或 Ollama）

# ==========================================
# 🔧 CONFIGURATION / 配置区域
# ==========================================

# 1. Hint Server Configuration
HINT_URL="http://localhost:5000/hint"

# 2. LLM Engine Selection (vllm or ollama)
ENGINE="ollama"  # 或 "vllm"

# 3. vLLM Configuration (如果使用 vllm)
VLLM_URL="http://localhost:8000/v1"
VLLM_MODEL="default"

# 4. Ollama Configuration (如果使用 ollama)
OLLAMA_URL="http://localhost:11434"
OLLAMA_MODEL="llama2"  # 确保已通过 'ollama pull llama2' 下载

# 5. Generation Parameters
PROMPT="Tell me a short story about network optimization."
MAX_TOKENS=200
TEMPERATURE=0.7

# ==========================================
# END CONFIGURATION
# ==========================================

# Ensure we are in the script's directory
cd "$(dirname "$0")" || exit

# Python detection
PYTHON_EXEC=$(which python3)
if [ -z "$PYTHON_EXEC" ]; then
    echo "❌ Error: 'python3' not found."
    exit 1
fi

echo "=================================================="
echo "   🚀 TokenFlow - Real LLM Client"
echo "=================================================="
echo ""
echo "📋 Configuration:"
echo "   Engine: $ENGINE"
echo "   Hint Server: $HINT_URL"
if [ "$ENGINE" = "vllm" ]; then
    echo "   vLLM URL: $VLLM_URL"
    echo "   vLLM Model: $VLLM_MODEL"
elif [ "$ENGINE" = "ollama" ]; then
    echo "   Ollama URL: $OLLAMA_URL"
    echo "   Ollama Model: $OLLAMA_MODEL"
fi
echo "   Prompt: $PROMPT"
echo ""

# Check if Hint Server is accessible
echo "🔍 Checking Hint Server..."
if curl -s "$HINT_URL" > /dev/null 2>&1; then
    echo "✅ Hint Server is accessible"
else
    echo "⚠️  Warning: Hint Server may not be running at $HINT_URL"
    echo "   You can start it with: sudo bash run_demo.sh"
    echo "   Or run with --disable-rate-limit to test without rate limiting"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Build command
CMD="$PYTHON_EXEC real_llm_client.py"
CMD="$CMD --engine $ENGINE"
CMD="$CMD --hint-url $HINT_URL"
CMD="$CMD --prompt \"$PROMPT\""
CMD="$CMD --max-tokens $MAX_TOKENS"
CMD="$CMD --temperature $TEMPERATURE"

if [ "$ENGINE" = "vllm" ]; then
    CMD="$CMD --vllm-url $VLLM_URL"
    CMD="$CMD --vllm-model $VLLM_MODEL"
elif [ "$ENGINE" = "ollama" ]; then
    CMD="$CMD --ollama-url $OLLAMA_URL"
    CMD="$CMD --ollama-model $OLLAMA_MODEL"
fi

echo "🚀 Starting Real LLM Client..."
echo ""
eval $CMD

