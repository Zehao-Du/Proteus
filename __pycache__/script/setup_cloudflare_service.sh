#!/bin/bash
# 将 Cloudflare Tunnel 设置为系统服务，确保断开 SSH 后仍能运行

set -e

CONFIG_FILE="$HOME/.cloudflared/config.yml"
TUNNEL_NAME="open-webui"
SERVICE_FILE="/etc/systemd/system/cloudflared-tunnel.service"

echo "🔧 设置 Cloudflare Tunnel 为系统服务..."

# 检查 root 权限
if [ "$EUID" -ne 0 ]; then 
    echo "❌ 需要 root 权限"
    echo "   请使用: sudo bash setup_cloudflare_service.sh"
    exit 1
fi

# 获取实际运行用户（不是 root）
if [ "$EUID" -eq 0 ]; then
    # 如果是 root，尝试从环境变量或当前登录用户获取
    if [ -n "$SUDO_USER" ]; then
        REAL_USER="$SUDO_USER"
    else
        # 尝试从当前登录会话获取
        REAL_USER=$(who am i | awk '{print $1}' | head -1)
        if [ -z "$REAL_USER" ]; then
            REAL_USER="argustest"  # 默认用户
        fi
    fi
else
    REAL_USER="$USER"
fi

HOME_DIR=$(eval echo ~$REAL_USER)
CONFIG_FILE="$HOME_DIR/.cloudflared/config.yml"

echo "   检测到用户: $REAL_USER"
echo "   主目录: $HOME_DIR"

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    echo "   请先运行: bash cloudflare_tunnel_setup.sh"
    exit 1
fi

# 获取 cloudflared 路径
CLOUDFLARED_PATH=$(which cloudflared)
if [ -z "$CLOUDFLARED_PATH" ]; then
    echo "❌ 找不到 cloudflared 命令"
    exit 1
fi

# 使用实际用户
USER="$REAL_USER"

echo "   用户: $USER"
echo "   配置文件: $CONFIG_FILE"
echo "   cloudflared: $CLOUDFLARED_PATH"

# 创建 systemd 服务文件
echo ""
echo "📄 创建 systemd 服务文件..."
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Cloudflare Tunnel for Open WebUI
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME_DIR
ExecStart=$CLOUDFLARED_PATH tunnel --config $CONFIG_FILE run $TUNNEL_NAME
Restart=always
RestartSec=10
StandardOutput=append:$HOME_DIR/cloudflare_tunnel.log
StandardError=append:$HOME_DIR/cloudflare_tunnel.log

[Install]
WantedBy=multi-user.target
EOF

echo "✅ 服务文件已创建: $SERVICE_FILE"

# 停止现有的 nohup 进程
echo ""
echo "🛑 停止现有的 Tunnel 进程..."
pkill -f "cloudflared tunnel.*$TUNNEL_NAME" 2>/dev/null || true
sleep 2

# 重新加载 systemd
echo ""
echo "🔄 重新加载 systemd..."
systemctl daemon-reload

# 启用服务
echo ""
echo "✅ 启用服务（开机自启）..."
systemctl enable cloudflared-tunnel.service

# 启动服务
echo ""
echo "🚀 启动服务..."
systemctl start cloudflared-tunnel.service

sleep 3

# 检查状态
echo ""
echo "📊 服务状态:"
systemctl status cloudflared-tunnel.service --no-pager -l | head -15

echo ""
echo "✅ 设置完成！"
echo ""
echo "📋 常用命令:"
echo "   查看状态: sudo systemctl status cloudflared-tunnel"
echo "   查看日志: tail -f $HOME_DIR/cloudflare_tunnel.log"
echo "   重启服务: sudo systemctl restart cloudflared-tunnel"
echo "   停止服务: sudo systemctl stop cloudflared-tunnel"
echo ""
echo "🌐 访问地址: https://riverli1616.uk"
echo ""
echo "💡 现在即使断开 SSH 连接，Tunnel 也会继续运行！"

