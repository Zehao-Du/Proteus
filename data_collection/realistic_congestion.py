#!/usr/bin/env python3
"""
Realistic Network Chaos Maker
基于学术界标准场景设计的网络故障注入脚本。
覆盖：Bufferbloat, Cellular Trace (Brownian Motion), Policer, Stochastic Loss.
"""
import time
import subprocess
import random
import math
import sys

# ================= 配置 =================
INTERFACE = "eth0"  # 请修改为你的实际网卡 (如 wlan0, ens33)
# ========================================

def run_cmd(cmd):
    # 使用 change 原子操作，避免断连
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def apply_netem(rate=None, delay=None, jitter=0, loss=0, limit=None):
    """
    构造并应用 TC NetEm 命令。
    """
    params = []
    if rate:
        # rate: 带宽 limit: 队列长度 (决定是丢包还是延迟)
        # 默认 limit 设大一点(3000)以模拟 Bufferbloat
        limit_val = limit if limit else 3000
        params.append(f"rate {rate}mbit limit {limit_val}")
    
    if delay:
        params.append(f"delay {delay}ms {jitter}ms distribution normal")
    
    if loss > 0:
        params.append(f"loss {loss}%")

    if not params:
        return

    param_str = " ".join(params)
    
    # 优先尝试 change，失败则 add
    cmd = f"sudo tc qdisc change dev {INTERFACE} root netem {param_str}"
    if subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        run_cmd(f"sudo tc qdisc add dev {INTERFACE} root netem {param_str}")

def clean_net():
    print(">>> [Cleanup] 恢复网络规则...")
    run_cmd(f"sudo tc qdisc del dev {INTERFACE} root")

# ================= 场景 1: Bufferbloat (排队论经典场景) =================
def scenario_bufferbloat():
    """
    理论: Queueing Delay = QueueLength / ServiceRate
    模拟: 带宽逐渐减少，而队列深度(limit)很大。
    特征: RTT 会呈现完美的线性上升，这是 LSTM 最容易捕捉的特征。
    """
    print(">>> 场景: Bufferbloat (RTT 线性爬升)")
    # 从 30Mbps 缓慢降到 2Mbps，持续 20秒
    start_rate = 30
    end_rate = 2
    duration = 20
    steps = 40 # 0.5s per step
    
    for i in range(steps):
        # 线性插值
        current_rate = start_rate - (start_rate - end_rate) * (i / steps)
        # Limit 很大 (5000包)，保证不丢包只排队
        apply_netem(rate=f"{current_rate:.2f}", delay=20, limit=5000)
        time.sleep(duration / steps)

# ================= 场景 2: Cellular Mobility (LTE 模拟) =================
def scenario_cellular_trace():
    """
    理论: 布朗运动 / Random Walk
    模拟: 移动网络下的带宽波动，不会瞬间跳变，而是连续波动。
    特征: 带宽和 RTT 有很强的自相关性 (Autocorrelation)。
    """
    print(">>> 场景: Cellular Mobility (带宽随机游走)")
    current_rate = 15.0
    duration = 30
    
    for _ in range(duration * 5): # 5Hz update
        # 随机波动 -2 ~ +2 Mbps
        delta = random.uniform(-2.0, 2.0)
        current_rate += delta
        # 限制范围 1Mbps ~ 50Mbps
        current_rate = max(1.0, min(50.0, current_rate))
        
        # 蜂窝网络通常伴随较大的 Jitter
        apply_netem(rate=f"{current_rate:.1f}", delay=40, jitter=15)
        time.sleep(0.2)

# ================= 场景 3: Policer / Token Bucket (ISP 限速) =================
def scenario_policer():
    """
    理论: 令牌桶算法 (Token Bucket)
    模拟: 突发流量允许通过，桶空了之后强制限速。
    特征: 吞吐量呈现“方波”或“锯齿波”，RTT 会出现周期性脉冲。
    """
    print(">>> 场景: ISP Policer (脉冲式限速)")
    # 模拟 5 个周期的令牌桶填充与耗尽
    for _ in range(5):
        # Phase 1: Burst (令牌充足) - 50Mbps, 低延迟
        apply_netem(rate=50, delay=10)
        time.sleep(2)
        
        # Phase 2: Capped (令牌耗尽) - 2Mbps, 强制排队或丢包
        # 这里 limit 设小一点，模拟 Policer 直接丢包
        apply_netem(rate=2, delay=10, limit=50) 
        time.sleep(3)

# ================= 场景 4: Deep Loss (弱信号) =================
def scenario_weak_signal():
    """
    理论: 物理层误码
    模拟: 带宽尚可，但随机丢包率高。
    特征: TCP 吞吐量下降，但 RTT 不一定升高（没有排队）。
    模型挑战: 区分“拥塞丢包”和“随机丢包”。
    """
    print(">>> 场景: Weak Signal (高随机丢包)")
    apply_netem(rate=20, delay=30, loss=5) # 5% 丢包
    time.sleep(15)
    
    apply_netem(rate=20, delay=30, loss=15) # 15% 丢包 (严重)
    time.sleep(10)

# ================= 主循环 =================
if __name__ == "__main__":
    print(f"🔥 Starting Realistic Chaos Engine on {INTERFACE}...")
    print("理论支撑: Bufferbloat, Brownian Motion, Token Bucket")
    
    clean_net()
    try:
        while True:
            # 随机选择一种物理场景，而不是随机生成参数
            scenario = random.choice([
                scenario_bufferbloat,
                scenario_cellular_trace,
                scenario_policer,
                scenario_weak_signal,
                # 偶尔恢复正常，让模型学习 baseline
                lambda: (print(">>> 场景: Normal Network"), clean_net(), time.sleep(10))
            ])
            
            # 执行场景
            if callable(scenario):
                scenario()
            else:
                scenario[0]() # lambda case
            
            # 场景间短暂休息
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 停止实验，恢复网络...")
        clean_net()