#!/bin/bash
# 网络损伤设置脚本
# 使用 Linux tc (Traffic Control) 工具模拟网络拥塞

INTERFACE="lo"  # 本地回环接口
ACTION="${1:-setup}"  # setup 或 cleanup

if [ "$ACTION" == "setup" ]; then
    echo "🔧 设置网络损伤..."
    
    # 检查是否已有规则
    if tc qdisc show dev $INTERFACE | grep -q "netem"; then
        echo "⚠️  检测到已有网络损伤规则，先清理..."
        sudo tc qdisc del dev $INTERFACE root 2>/dev/null || true
    fi
    
    # 设置网络损伤参数
    LOSS="${2:-5}"      # 丢包率 (%)
    DELAY="${3:-50}"    # 延迟 (ms)
    JITTER="${4:-10}"   # 抖动 (ms)
    
    echo "   丢包率: ${LOSS}%"
    echo "   延迟: ${DELAY}ms"
    echo "   抖动: ±${JITTER}ms"
    
    # 添加网络损伤规则
    sudo tc qdisc add dev $INTERFACE root netem \
        loss ${LOSS}% \
        delay ${DELAY}ms ${JITTER}ms \
        distribution normal
    
    echo "✅ 网络损伤已设置"
    echo "   使用 'sudo tc qdisc show dev $INTERFACE' 查看当前规则"
    echo "   使用 './setup_network_impairment.sh cleanup' 清理规则"
    
elif [ "$ACTION" == "cleanup" ]; then
    echo "🧹 清理网络损伤规则..."
    sudo tc qdisc del dev $INTERFACE root 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✅ 网络损伤规则已清理"
    else
        echo "⚠️  没有找到需要清理的规则（可能已经清理过了）"
    fi
    
elif [ "$ACTION" == "status" ]; then
    echo "📊 当前网络损伤状态:"
    tc qdisc show dev $INTERFACE
    
else
    echo "用法: $0 {setup|cleanup|status} [loss%] [delay_ms] [jitter_ms]"
    echo ""
    echo "示例:"
    echo "  $0 setup 10 100 20    # 10% 丢包，100ms 延迟，±20ms 抖动"
    echo "  $0 cleanup             # 清理所有规则"
    echo "  $0 status              # 查看当前状态"
fi

