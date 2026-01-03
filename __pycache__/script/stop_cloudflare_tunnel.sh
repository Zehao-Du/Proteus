#!/bin/bash
# 停止 Cloudflare Tunnel 服务

TUNNEL_NAME="open-webui"

echo "🛑 停止 Cloudflare Tunnel..."

if pgrep -f "cloudflared tunnel.*$TUNNEL_NAME" > /dev/null; then
    pkill -f "cloudflared tunnel.*$TUNNEL_NAME"
    sleep 2
    
    if ! pgrep -f "cloudflared tunnel.*$TUNNEL_NAME" > /dev/null; then
        echo "✅ Tunnel 已停止"
    else
        echo "⚠️  Tunnel 仍在运行，强制停止..."
        pkill -9 -f "cloudflared tunnel.*$TUNNEL_NAME"
    fi
else
    echo "ℹ️  Tunnel 未运行"
fi


