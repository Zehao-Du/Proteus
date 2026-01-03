#!/bin/bash
# 启动 Cloudflare Tunnel 服务

CONFIG_FILE="$HOME/.cloudflared/config.yml"
TUNNEL_NAME="open-webui"
LOG_FILE="$HOME/cloudflare_tunnel.log"

echo "🚀 启动 Cloudflare Tunnel..."

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    echo "请先运行: bash cloudflare_tunnel_setup.sh"
    exit 1
fi

# 检查隧道是否已在运行
if pgrep -f "cloudflared tunnel.*$TUNNEL_NAME" > /dev/null; then
    echo "⚠️  Tunnel 已在运行中"
    echo "停止现有进程..."
    pkill -f "cloudflared tunnel.*$TUNNEL_NAME"
    sleep 2
fi

# 启动隧道（后台运行）
echo "📡 启动隧道: $TUNNEL_NAME"
nohup cloudflared tunnel --config "$CONFIG_FILE" run "$TUNNEL_NAME" > "$LOG_FILE" 2>&1 &

sleep 3

# 检查是否启动成功
if pgrep -f "cloudflared tunnel.*$TUNNEL_NAME" > /dev/null; then
    echo "✅ Tunnel 启动成功！"
    echo "📋 日志文件: $LOG_FILE"
    echo "🌐 访问地址: https://riverli1616.uk"
    echo ""
    echo "查看日志: tail -f $LOG_FILE"
    echo "停止隧道: pkill -f 'cloudflared tunnel.*$TUNNEL_NAME'"
else
    echo "❌ Tunnel 启动失败，请查看日志: $LOG_FILE"
    tail -20 "$LOG_FILE"
    exit 1
fi


