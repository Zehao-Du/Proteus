#!/usr/bin/env python3
"""
Chaos Maker for Loopback Interface (lo)

用于在本地测试时模拟网络延迟和丢包。
当 vLLM 服务和客户端在同一台机器上时，需要对 lo 接口注入故障。

Usage:
    sudo python3 chaos_lo.py
    sudo python3 chaos_lo.py --delay 100 --loss 5
"""

import argparse
import subprocess
import time
import signal
import sys

INTERFACE = "lo"  # Loopback interface for localhost traffic

def run_cmd(cmd):
    """执行 Shell 命令"""
    print(f"  $ {cmd}")
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def apply_netem(delay=0, jitter=0, loss=0):
    """应用 NetEm 规则"""
    params = []
    if delay > 0:
        params.append(f"delay {delay}ms {jitter}ms distribution normal")
    if loss > 0:
        params.append(f"loss {loss}%")
    
    param_str = " ".join(params) if params else "delay 0ms"
    
    # 先尝试 change，失败则 add
    cmd_change = f"sudo tc qdisc change dev {INTERFACE} root netem {param_str}"
    cmd_add = f"sudo tc qdisc add dev {INTERFACE} root netem {param_str}"
    
    ret = subprocess.call(cmd_change, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if ret != 0:
        run_cmd(cmd_add)
    
    print(f"🌪️  Applied: Delay={delay}±{jitter}ms, Loss={loss}%")

def clean():
    """清理规则"""
    print(f"\n🧹 Cleaning up {INTERFACE} rules...")
    run_cmd(f"sudo tc qdisc del dev {INTERFACE} root")
    print("✅ Network restored to normal")

def signal_handler(sig, frame):
    clean()
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Chaos injection for loopback interface")
    parser.add_argument("--delay", type=int, default=50, help="Base delay in ms (default: 50)")
    parser.add_argument("--jitter", type=int, default=20, help="Jitter in ms (default: 20)")
    parser.add_argument("--loss", type=float, default=5.0, help="Packet loss %% (default: 5.0)")
    parser.add_argument("--duration", type=int, default=0, help="Duration in seconds (0 = infinite)")
    args = parser.parse_args()
    
    # Handle Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("=" * 50)
    print("  Chaos Maker for Loopback (lo) Interface")
    print("=" * 50)
    print(f"  Interface: {INTERFACE}")
    print(f"  Delay: {args.delay}±{args.jitter}ms")
    print(f"  Loss: {args.loss}%")
    print("=" * 50)
    print()
    
    # Clean first
    clean()
    time.sleep(0.5)
    
    # Apply chaos
    apply_netem(delay=args.delay, jitter=args.jitter, loss=args.loss)
    
    print("\n⏳ Chaos active. Press Ctrl+C to stop and restore network.\n")
    
    if args.duration > 0:
        time.sleep(args.duration)
        clean()
    else:
        # Run forever until Ctrl+C
        while True:
            time.sleep(1)

if __name__ == "__main__":
    main()

