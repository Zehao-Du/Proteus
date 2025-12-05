#!/usr/bin/python3
from bcc import BPF
import time

# ==========================================
# 1. eBPF 内核代码 (C Language)
# ==========================================
bpf_text = """
#include <uapi/linux/ptrace.h>
#include <net/sock.h>
#include <bcc/proto.h>
#include <linux/tcp.h>

// 屏蔽某些可能引起冲突的宏警告
#undef __HAVE_BUILTIN_BSWAP32__
#undef __HAVE_BUILTIN_BSWAP64__
#undef __HAVE_BUILTIN_BSWAP16__

struct data_t {
    u32 pid;
    u32 saddr;
    u32 daddr;
    u16 lport;
    u16 dport;
    u32 rtt;
};

BPF_PERF_OUTPUT(events);

int trace_tcp_rcv(struct pt_regs *ctx, struct sock *sk)
{
    u32 pid = bpf_get_current_pid_tgid() >> 32;

    // !!! 修复点：直接强制类型转换，避开内核宏的坑 !!!
    struct tcp_sock *ts = (struct tcp_sock *)sk;
    
    // 获取 IP 和端口
    u32 saddr = sk->sk_rcv_saddr;
    u32 daddr = sk->sk_daddr;
    u16 lport = sk->sk_num;
    u16 dport = sk->sk_dport;
    
    // 获取 RTT
    // 如果 srtt_us 为 0，说明连接刚建立还没有 RTT 数据，或者不是 TCP
    u32 srtt = ts->srtt_us >> 3;

    if (srtt == 0) {
        return 0; // 过滤掉没有 RTT 数据的包
    }

    struct data_t data = {};
    data.pid = pid;
    data.saddr = saddr;
    data.daddr = daddr;
    data.lport = lport;
    data.dport = dport; // 注意：这里通常是网络字节序
    data.rtt = srtt;

    events.perf_submit(ctx, &data, sizeof(data));

    return 0;
}
"""

# ==========================================
# 2. Python 用户态代码
# ==========================================

print("正在编译 eBPF 代码，请稍候...")
try:
    b = BPF(text=bpf_text)
except Exception as e:
    print("\n[错误] 编译失败！")
    print("这通常是 WSL2 内核头文件路径或版本不匹配导致的。")
    print("错误详情:\n", e)
    exit(1)

print("eBPF 加载成功！正在 Hook tcp_rcv_established...")
try:
    b.attach_kprobe(event="tcp_rcv_established", fn_name="trace_tcp_rcv")
except Exception as e:
    print("\n[错误] 挂载失败！可能是函数名在当前内核中改变了。")
    # 备选方案：尝试旧内核函数名
    try:
        print("尝试挂载 tcp_v4_rcv...")
        b.attach_kprobe(event="tcp_v4_rcv", fn_name="trace_tcp_rcv")
    except:
        print(e)
        exit(1)

print("%-10s %-15s %-6s -> %-15s %-6s %-10s" % ("PID", "SRC_IP", "PORT", "DST_IP", "PORT", "RTT(us)"))

def inet_ntoa(addr):
    import socket
    import struct
    try:
        return socket.inet_ntoa(struct.pack("I", addr))
    except:
        return "0.0.0.0"

def print_event(cpu, data, size):
    event = b["events"].event(data)
    
    # 1. 解析 IP 地址
    s_ip = inet_ntoa(event.saddr)
    d_ip = inet_ntoa(event.daddr)
    
    # === 新增：过滤掉本机流量 (127.0.0.1) ===
    if s_ip == "127.0.0.1" or d_ip == "127.0.0.1":
        return
    # ======================================

    # 2. 解析端口
    import socket
    final_dport = socket.ntohs(event.dport)
    
    print("%-10d %-15s %-6d -> %-15s %-6d %-10d" % (
        event.pid,
        s_ip,
        event.lport,
        d_ip,
        final_dport,
        event.rtt
    ))

b["events"].open_perf_buffer(print_event)

print("开始监听... (按 Ctrl+C 停止)")
while True:
    try:
        b.perf_buffer_poll()
    except KeyboardInterrupt:
        print("\n停止监听。")
        exit()