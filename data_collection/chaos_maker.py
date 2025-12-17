import os
import time
import random
import subprocess
import datetime

# =================配置区域=================
INTERFACE = "eth0"  # 请确认你的网卡名称 (ip addr 查看)
LOG_FILE = "chaos_log.log"
# ==========================================

def run_cmd(cmd):
    """执行 Shell 命令，隐藏输出以免刷屏"""
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def apply_netem(delay=0, jitter=0, loss=0, rate=None, duplicate=0):
    """
    核心函数：应用 NetEm 规则
    使用 'change' 而不是 'del + add' 以保持连接平滑
    """
    # 构造 netem 参数字符串
    params = []
    if delay > 0:
        # delay <time> <jitter> distribution normal
        params.append(f"delay {delay}ms {jitter}ms distribution normal")
    
    if loss > 0:
        params.append(f"loss {loss}%")
        
    if duplicate > 0:
        params.append(f"duplicate {duplicate}%")
    
    if rate:
        # rate <speed>kbit limit <packets>
        # limit 是缓冲区大小，设置小了会丢包，设置大了会增加 RTT (Bufferbloat)
        # 这里故意设置大一点的 limit 以观察 RTT 上升
        params.append(f"rate {rate}mbit limit 5000")

    param_str = " ".join(params)
    
    # 构建完整命令
    # 1. 尝试 change (如果规则已存在)
    cmd_change = f"sudo tc qdisc change dev {INTERFACE} root netem {param_str}"
    # 2. 尝试 add (如果规则不存在，即第一次运行)
    cmd_add = f"sudo tc qdisc add dev {INTERFACE} root netem {param_str}"
    
    # 优先尝试 change
    ret = subprocess.call(cmd_change, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if ret != 0:
        # 如果 change 失败（通常是因为规则不存在），则执行 add
        run_cmd(cmd_add)
        
    log_msg = f"[{datetime.datetime.now()}] SET: Delay={delay}±{jitter}ms, Loss={loss}%, Rate={rate or 'Unlim'}Mbps"
    print(f">>> {log_msg}")
    with open(LOG_FILE, "a") as f:
        f.write(log_msg + "\n")

def clean_net():
    print(f"[{datetime.datetime.now()}] >>> 清理网络规则 (恢复正常)")
    run_cmd(f"sudo tc qdisc del dev {INTERFACE} root")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.datetime.now()}] CLEAN: Network Restored\n")

# ================= 场景定义 =================

def scenario_normal():
    # 正常网络也有一点点底噪
    apply_netem(delay=0, jitter=0, loss=0, rate=0)

def scenario_jittery_wifi():
    # 模拟不稳定的 WiFi: 延迟波动大，无丢包
    base = random.randint(20, 50)
    jitter = random.randint(10, 30)
    apply_netem(delay=base, jitter=jitter, loss=0)

def scenario_weak_signal():
    # 模拟弱信号: 高丢包
    loss = random.randint(5, 15)
    apply_netem(delay=50, jitter=10, loss=loss)

def scenario_congestion():
    # 模拟拥塞 (Bufferbloat): 带宽受限 + 延迟
    # 这对 RTT 预测最具挑战性
    rate = random.randint(1, 10) # 1-10 Mbps
    apply_netem(delay=20, jitter=5, rate=rate)

def scenario_heavy_load():
    # 混合恶劣环境
    apply_netem(delay=150, jitter=40, loss=5, rate=2)

if __name__ == "__main__":
    print(f"开始高级故障注入... 目标网卡: {INTERFACE}")
    print("请确保 smart_agent.py 正在运行...")
    
    # 确保从干净状态开始
    clean_net()
    
    try:
        # 1. 预热 (基线数据)
        print("--- 预热阶段 (30s) ---")
        scenario_normal()
        time.sleep(30)

        # 2. 循环训练场景
        scenarios = [
            (scenario_jittery_wifi, "WiFi Jitter"),
            (scenario_weak_signal, "Packet Loss"),
            (scenario_congestion, "Bandwidth Congestion"),
            (scenario_heavy_load, "Heavy Load (Mixed)"),
            (scenario_normal, "Normal Recovery")
        ]

        while True:
            # 随机选择场景
            func, name = random.choice(scenarios)
            print(f"\n--- 切换场景: {name} ---")
            
            # 执行场景
            func()
            
            # 持续时间随机化 (20s - 50s)
            # 时间不宜太短，否则 TCP 还没调整完状态就变了，模型学不到稳态
            duration = random.randint(20, 50)
            
            # 倒计时显示
            for s in range(duration, 0, -5):
                print(f"保持 {s} 秒...", end="\r")
                time.sleep(5)
            print(" " * 20, end="\r")

    except KeyboardInterrupt:
        print("\n检测到退出信号...")
    finally:
        clean_net()
        print("实验结束，规则已清理。")