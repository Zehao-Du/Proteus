#!/usr/bin/python3
from bcc import BPF
import socket
import struct
import time

# ==========================================
# 1. eBPF 内核代码
# ==========================================
bpf_text = """
#include <uapi/linux/ptrace.h>
#include <net/sock.h>
#include <bcc/proto.h>
#include <linux/tcp.h>

// 屏蔽宏重定义警告
#undef __HAVE_BUILTIN_BSWAP32__
#undef __HAVE_BUILTIN_BSWAP64__
#undef __HAVE_BUILTIN_BSWAP16__

struct data_t {
    u32 pid;
    u32 saddr;
    u32 daddr;
    u16 lport;
    u16 dport;
    u64 ts_us; // 记录发生时间
};

BPF_PERF_OUTPUT(events);

// Hook 函数：当 TCP 发生重传时触发
int trace_retransmit(struct pt_regs *ctx, struct sock *sk)
{
    u32 pid = bpf_get_current_pid_tgid() >> 32;

    // 强制类型转换，避开复杂宏
    struct tcp_sock *ts = (struct tcp_sock *)sk;
    
    // 获取 IP 和端口
    u32 saddr = sk->sk_rcv_saddr;
    u32 daddr = sk->sk_daddr;
    u16 lport = sk->sk_num;
    u16 dport = sk->sk_dport;

    // 过滤掉本地流量 (127.0.0.1 = 0x0100007F in hex, little endian)
    if (saddr == 0x0100007F || daddr == 0x0100007F) {
        return 0;
    }

    struct data_t data = {};
    data.pid = pid;
    data.saddr = saddr;
    data.daddr = daddr;
    data.lport = lport;
    data.dport = dport;
    data.ts_us = bpf_ktime_get_ns() / 1000;

    events.perf_submit(ctx, &data, sizeof(data));

    return 0;
}
"""

# ==========================================
# 2. Python 用户态代码
# ==========================================

print("正在编译重传监控工具...")
b = BPF(text=bpf_text)

# 核心 Hook 点：tcp_retransmit_skb
# Linux 内核在重传 TCP 包时会调用这个函数
b.attach_kprobe(event="tcp_retransmit_skb", fn_name="trace_retransmit")

print("eBPF 重传监控已启动！等待异常发生...")
print("%-10s %-15s %-6s -> %-15s %-6s %-10s" % ("TIME", "SRC_IP", "PORT", "DST_IP", "PORT", "TYPE"))

def inet_ntoa(addr):
    try:
        return socket.inet_ntoa(struct.pack("I", addr))
    except:
        return "0.0.0.0"

def print_event(cpu, data, size):
    event = b["events"].event(data)
    final_dport = socket.ntohs(event.dport)
    
    # 打印红色的 "RETRANS" 看起来更像警告
    print("%-10d %-15s %-6d -> %-15s %-6d \033[91mRETRANS\033[0m" % (
        event.ts_us % 100000, # 简化的时间戳
        inet_ntoa(event.saddr),
        event.lport,
        inet_ntoa(event.daddr),
        final_dport
    ))

b["events"].open_perf_buffer(print_event)

while True:
    try:
        b.perf_buffer_poll()
    except KeyboardInterrupt:
        exit()