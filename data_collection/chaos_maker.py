import os
import time
import random

# 确保安装了 tc
# 注意：WSL2 中网卡通常是 eth0，请用 ip addr 确认一下
INTERFACE = "eth0" 

def run_cmd(cmd):
    print(f"执行命令: {cmd}")
    os.system(cmd)

def clean_net():
    print(">>> 清理网络规则...")
    run_cmd(f"sudo tc qdisc del dev {INTERFACE} root 2> /dev/null")

def set_normal():
    print(">>> [状态：正常] 网络通畅")
    clean_net()

def set_delay(ms):
    print(f">>> [状态：高延迟] 增加 {ms}ms 延迟")
    clean_net()
    run_cmd(f"sudo tc qdisc add dev {INTERFACE} root netem delay {ms}ms")

def set_loss(percent):
    print(f">>> [状态：丢包] 增加 {percent}% 丢包")
    clean_net()
    run_cmd(f"sudo tc qdisc add dev {INTERFACE} root netem loss {percent}%")

def set_mixed(ms, percent):
    print(f">>> [状态：混合故障] 延迟 {ms}ms + 丢包 {percent}%")
    clean_net()
    run_cmd(f"sudo tc qdisc add dev {INTERFACE} root netem delay {ms}ms loss {percent}%")

if __name__ == "__main__":
    print("开始自动制造故障... (请确保另一个窗口正在运行 smart_agent.py)")
    
    try:
        # 1. 先跑 60秒 正常数据 (建立基线)
        set_normal()
        time.sleep(60)

        # 2. 循环制造故障
        for i in range(5):
            # 随机选择一种故障模式
            mode = random.choice(['delay', 'loss', 'mixed', 'normal'])
            duration = random.randint(30, 60) # 持续 30-60 秒

            if mode == 'normal':
                set_normal()
            elif mode == 'delay':
                delay_val = random.randint(50, 300) # 50ms - 300ms
                set_delay(delay_val)
            elif mode == 'loss':
                loss_val = random.randint(5, 20) # 5% - 20%
                set_loss(loss_val)
            elif mode == 'mixed':
                set_mixed(random.randint(50, 200), random.randint(1, 10))
            
            # 等待故障持续
            print(f"保持 {duration} 秒...")
            time.sleep(duration)
            
            # 每次故障后，恢复正常 10 秒，让数据有个回落
            set_normal()
            time.sleep(10)

    except KeyboardInterrupt:
        clean_net()
        print("\n实验结束，网络已恢复。")
    finally:
        clean_net()