#!/bin/bash
# =============================================================================
# eBPF-TokenFlow A/B Experiment Launcher
# 网络感知调度 A/B 实验启动脚本
# =============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

PROJECT_ROOT="/home/v-boxiuli/eBPF-TokenFlow"
VLLM_PORT=8000
HINT_PORT=5000

print_header() {
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║   ${CYAN}eBPF-TokenFlow: Network-Aware LLM Scheduling A/B Test${BLUE}     ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
}

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

show_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --rounds N        Number of experiment rounds (default: 5)"
    echo "  --max-tokens N    Base max tokens per user (default: 80)"
    echo "  --start-hint      Start the Hint Server"
    echo "  --start-vllm      Show vLLM startup command"
    echo "  -h, --help        Show this help message"
}

start_hint_server() {
    echo -e "\n${BLUE}[Starting Hint Server]${NC}"
    if check_service $HINT_PORT "Hint Server"; then
        echo -e "${YELLOW}Hint Server already running. Kill it first if you need to restart.${NC}"
        return 0
    fi
    
    cd $PROJECT_ROOT/demo
    source /home/v-boxiuli/miniconda3/etc/profile.d/conda.sh
    conda activate smartnet
    
    echo -e "${GREEN}Starting Multi-User Hint Server...${NC}"
    python3 multi_user_hint_server.py --port $HINT_PORT &
    sleep 3
    
    if check_service $HINT_PORT "Hint Server"; then
        echo -e "${GREEN}✅ Hint Server started successfully${NC}"
    else
        echo -e "${RED}❌ Failed to start Hint Server${NC}"
        exit 1
    fi
}

show_vllm_command() {
    echo -e "\n${BLUE}[vLLM Startup Command]${NC}"
    echo -e "${YELLOW}Run the following command in a separate terminal:${NC}"
    echo ""
    echo -e "${CYAN}cd $PROJECT_ROOT/vllm${NC}"
    echo -e "${CYAN}source /home/v-boxiuli/miniconda3/etc/profile.d/conda.sh && conda activate smartnet${NC}"
    echo -e "${CYAN}export VLLM_HINT_SERVER_URL=http://localhost:$HINT_PORT/hint${NC}"
    echo -e "${CYAN}export PYTHONPATH=\$(pwd):\$PYTHONPATH${NC}"
    echo -e "${CYAN}python -m vllm.entrypoints.openai.api_server \\${NC}"
    echo -e "${CYAN}    --model Qwen/Qwen3-4B-Instruct-2507 \\${NC}"
    echo -e "${CYAN}    --port $VLLM_PORT \\${NC}"
    echo -e "${CYAN}    --trust-remote-code \\${NC}"
    echo -e "${CYAN}    --gpu-memory-utilization 0.4 \\${NC}"
    echo -e "${CYAN}    --max-model-len 2048${NC}"
    echo ""
}

run_experiment() {
    local rounds=$1
    local max_tokens=$2
    
    echo -e "\n${BLUE}[Pre-flight Checks]${NC}"
    
    if ! check_service $VLLM_PORT "vLLM"; then
        echo -e "${RED}❌ vLLM is not running. Please start it first.${NC}"
        show_vllm_command
        exit 1
    fi
    
    if ! check_service $HINT_PORT "Hint Server"; then
        echo -e "${YELLOW}Starting Hint Server...${NC}"
        start_hint_server
    fi
    
    echo -e "\n${BLUE}[Running A/B Experiment]${NC}"
    echo -e "   Rounds: ${CYAN}$rounds${NC}"
    echo -e "   Max Tokens: ${CYAN}$max_tokens${NC}"
    
    cd $PROJECT_ROOT/demo
    source /home/v-boxiuli/miniconda3/etc/profile.d/conda.sh
    conda activate smartnet
    
    python3 concurrent_ab_experiment_v2.py \
        --vllm-url "http://localhost:$VLLM_PORT" \
        --hint-url "http://localhost:$HINT_PORT/hint" \
        --rounds $rounds \
        --max-tokens $max_tokens
    
    echo -e "\n${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                   Experiment Complete!                       ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
}

# 默认参数
ROUNDS=5
MAX_TOKENS=80
START_HINT=false
START_VLLM=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --rounds)
            ROUNDS="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --start-hint)
            START_HINT=true
            shift
            ;;
        --start-vllm)
            START_VLLM=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# 主程序
print_header

if [ "$START_HINT" = true ]; then
    start_hint_server
    exit 0
fi

if [ "$START_VLLM" = true ]; then
    show_vllm_command
    exit 0
fi

run_experiment $ROUNDS $MAX_TOKENS

