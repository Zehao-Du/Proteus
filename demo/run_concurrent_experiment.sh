#!/bin/bash
# =============================================================================
# Concurrent A/B Experiment Launcher
# 并发 A/B 实验启动脚本
#
# 本脚本会启动所有必要的组件来运行网络感知调度的 A/B 实验
# =============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PROJECT_ROOT="/home/v-boxiuli/eBPF-TokenFlow"
VLLM_PORT=8000
HINT_PORT=5000

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Concurrent A/B Experiment Launcher${NC}"
echo -e "${BLUE}============================================${NC}"

# 检查是否需要启动服务
check_service() {
    local port=$1
    local name=$2
    if nc -z localhost $port 2>/dev/null; then
        echo -e "${GREEN}✅ $name is running on port $port${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  $name is NOT running on port $port${NC}"
        return 1
    fi
}

# 停止所有相关进程
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    pkill -f "multi_user_hint_server" 2>/dev/null || true
    echo -e "${GREEN}✅ Cleanup complete${NC}"
}

trap cleanup EXIT

# 主流程
main() {
    echo -e "\n${BLUE}[Step 1] Checking services...${NC}"
    
    # 检查 vLLM
    if ! check_service $VLLM_PORT "vLLM"; then
        echo -e "${RED}❌ Please start vLLM first:${NC}"
        echo "cd $PROJECT_ROOT && export VLLM_HINT_SERVER_URL=http://localhost:$HINT_PORT/hint"
        echo "python3 -m vllm.entrypoints.openai.api_server \\"
        echo "    --model Qwen/Qwen3-4B-Instruct-2507 \\"
        echo "    --port $VLLM_PORT \\"
        echo "    --trust-remote-code \\"
        echo "    --gpu-memory-utilization 0.4 \\"
        echo "    --max-model-len 2048"
        exit 1
    fi
    
    # 启动 Multi-User Hint Server
    echo -e "\n${BLUE}[Step 2] Starting Multi-User Hint Server...${NC}"
    cd $PROJECT_ROOT/demo
    python3 multi_user_hint_server.py --port $HINT_PORT &
    HINT_PID=$!
    echo -e "${GREEN}✅ Hint Server started (PID: $HINT_PID)${NC}"
    
    # 等待服务启动
    echo -e "\n${BLUE}[Step 3] Waiting for services...${NC}"
    sleep 3
    
    if ! check_service $HINT_PORT "Hint Server"; then
        echo -e "${RED}❌ Hint Server failed to start${NC}"
        exit 1
    fi
    
    # 运行实验
    echo -e "\n${BLUE}[Step 4] Running Concurrent A/B Experiment...${NC}"
    cd $PROJECT_ROOT/demo
    python3 concurrent_ab_experiment.py \
        --vllm-url "http://localhost:$VLLM_PORT" \
        --hint-url "http://localhost:$HINT_PORT/hint" \
        --rounds 5 \
        --max-tokens 100
    
    echo -e "\n${GREEN}============================================${NC}"
    echo -e "${GREEN}  Experiment Complete!${NC}"
    echo -e "${GREEN}============================================${NC}"
}

main "$@"

