#!/usr/bin/env python3
"""Smart Agent v2: Enhanced TCP Monitor for RTT Prediction.

Changes from v1:
1. Collects Throughput (Bytes/sec) and CWND (Congestion Window).
2. Generates supervised learning labels (Next_RTT) via 1-step buffering.
"""
import argparse
import csv
import statistics
import time
from collections import deque

from bcc import BPF

# ============================================================
# eBPF Kernel Program
# ============================================================

BPF_PROGRAM = r"""
#ifndef KBUILD_MODNAME
#define KBUILD_MODNAME "ebpf_tokenflow"
#endif

#ifndef __BPF_TRACING__
#define __BPF_TRACING__
#endif

// 1. 强制屏蔽段寄存器和 CPU 类型宏，防止 6.x 内核冲突
#undef __seg_gs
#define __seg_gs
#undef __seg_fs
#define __seg_fs
#undef __my_cpu_type
#define __my_cpu_type(var) typeof(var)

// 2. 预定义信号宏，防止信号数组越界报错
#undef _NSIG_WORDS
#define _NSIG_WORDS 4

// 3. 补全缺失的内核结构体定义，解决 sizeof(struct bpf_wq) 报错
struct bpf_wq { int x; };

#include <uapi/linux/ptrace.h>
#include <linux/types.h>

// BCC 现代版本已内置这些结构，无需手动定义
#ifndef BPF_PSEUDO_FUNC
#define BPF_PSEUDO_FUNC 4
#endif

#include <net/sock.h>
#include <bcc/proto.h>
#include <linux/tcp.h>
#include <linux/skbuff.h>

BPF_PERF_OUTPUT(rtt_events);
BPF_PERF_OUTPUT(retrans_events);

struct rtt_data_t {
    u32 rtt;
    u32 cwnd;
    u32 len;
};

struct retrans_data_t {
    u32 dummy;
};

// trace_tcp_rcv: Called when a valid TCP packet is received
// We add 'struct sk_buff *skb' to arguments to get packet length
int trace_tcp_rcv(struct pt_regs *ctx, struct sock *sk, struct sk_buff *skb)
{
    struct tcp_sock *ts = (struct tcp_sock *)sk;
    u32 srtt = ts->srtt_us >> 3;

    if (srtt == 0) return 0;

    struct rtt_data_t data = {};
    data.rtt = srtt;
    data.cwnd = ts->snd_cwnd; // Congestion Window
    
    // Safety check for skb access
    if (skb) {
        data.len = skb->len;  // Packet length in bytes
    } else {
        data.len = 0;
    }

    rtt_events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}

int trace_retransmit(struct pt_regs *ctx, struct sock *sk)
{
    struct retrans_data_t data = {};
    retrans_events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
"""

class SmartAgent:
    def __init__(self, interval: float, window: int, csv_path: str, max_samples: int):
        self.interval = interval
        self.window = window
        self.csv_path = csv_path
        self.max_samples = max_samples

        print("Compiling eBPF program...")
        self.bpf = BPF(text=BPF_PROGRAM)
        # Attach to kprobe
        self.bpf.attach_kprobe(event="tcp_rcv_established", fn_name="trace_tcp_rcv")
        self.bpf.attach_kprobe(event="tcp_retransmit_skb", fn_name="trace_retransmit")

        # Runtime buffers
        self.rtt_samples = []
        self.cwnd_samples = []
        self.total_bytes = 0
        self.retrans_count = 0
        
        # Rolling window for trend features
        self.window_buffer = deque(maxlen=self.window)
        
        # Buffer for Label Generation (Next RTT)
        self.last_interval_metrics = None

        self.bpf["rtt_events"].open_perf_buffer(self._handle_rtt)
        self.bpf["retrans_events"].open_perf_buffer(self._handle_retrans)

    # -------------------------
    # eBPF Callbacks
    # -------------------------
    def _handle_rtt(self, cpu, data, size):
        event = self.bpf["rtt_events"].event(data)
        if len(self.rtt_samples) < self.max_samples:
            self.rtt_samples.append(event.rtt)
            self.cwnd_samples.append(event.cwnd)
        
        # Accumulate throughput regardless of sample cap
        self.total_bytes += event.len

    def _handle_retrans(self, cpu, data, size):
        self.retrans_count += 1

    # -------------------------
    # Helpers
    # -------------------------
    def _percentile(self, values, pct):
        if not values: return 0
        k = (len(values) - 1) * pct
        f = int(k)
        c = min(f + 1, len(values) - 1)
        return values[f] + (values[c] - values[f]) * (k - f)

    def _aggregate_metrics(self):
        """Calculates stats for the current interval."""
        # 1. RTT Stats
        if not self.rtt_samples:
            avg_rtt = p95_rtt = min_rtt = max_rtt = 0
            avg_cwnd = 0
        else:
            sorted_rtt = sorted(self.rtt_samples)
            avg_rtt = int(statistics.fmean(sorted_rtt))
            p95_rtt = int(self._percentile(sorted_rtt, 0.95))
            min_rtt = sorted_rtt[0]
            max_rtt = sorted_rtt[-1]
            avg_cwnd = int(statistics.fmean(self.cwnd_samples))

        # 2. Throughput (Bytes per second)
        # Note: self.interval is the polling time
        throughput = int(self.total_bytes / self.interval)

        metrics = {
            "timestamp": int(time.time()),
            "avg_rtt_us": avg_rtt,
            "p95_rtt_us": p95_rtt,
            "min_rtt_us": min_rtt,
            "max_rtt_us": max_rtt,
            "avg_cwnd": avg_cwnd,
            "throughput_bps": throughput, # New Feature
            "retrans_count": self.retrans_count,
            "rtt_samples": len(self.rtt_samples),
        }

        # 3. Rolling Window Features
        self.window_buffer.append(metrics)
        if self.window_buffer:
            metrics["rolling_avg_rtt"] = int(statistics.fmean(m["avg_rtt_us"] for m in self.window_buffer))
            metrics["rolling_p95_rtt"] = int(statistics.fmean(m["p95_rtt_us"] for m in self.window_buffer))
            metrics["rolling_std_rtt"] = int(statistics.stdev([m["avg_rtt_us"] for m in self.window_buffer]) if len(self.window_buffer) > 1 else 0)
        else:
            metrics["rolling_avg_rtt"] = 0
            metrics["rolling_p95_rtt"] = 0
            metrics["rolling_std_rtt"] = 0

        return metrics

    def _reset_counters(self):
        self.rtt_samples = []
        self.cwnd_samples = []
        self.total_bytes = 0
        self.retrans_count = 0

    def run(self):
        print(f"Smart Agent Running. Interval={self.interval}s. Writing to {self.csv_path}")
        self._prepare_csv()
        
        try:
            while True:
                self._poll_events()
                
                # Get current stats (Features for time T)
                current_metrics = self._aggregate_metrics()
                
                # Logic: We write the ROW for time (T-1) now, 
                # using time (T)'s RTT as the TARGET Label.
                if self.last_interval_metrics is not None:
                    # Feature: State at T-1
                    # Label: RTT at T
                    target_next_rtt = current_metrics["avg_rtt_us"]
                    self._write_row(self.last_interval_metrics, target_next_rtt)
                    self._print_row(self.last_interval_metrics, target_next_rtt)
                
                # Store T to be written when T+1 arrives
                self.last_interval_metrics = current_metrics
                
                self._reset_counters()
                
        except KeyboardInterrupt:
            print("\nStopping collector...")
            # Note: The very last interval captured is lost here because 
            # we don't have a "next" interval to label it. This is acceptable.

    def _poll_events(self):
        start = time.time()
        # Ensure we poll for exactly 'interval' seconds
        while time.time() - start < self.interval:
            self.bpf.perf_buffer_poll(timeout=100)

    def _prepare_csv(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                # --- Features ---
                "avg_rtt_us", "p95_rtt_us", "min_rtt_us", "max_rtt_us",
                "avg_cwnd", "throughput_bps", "retrans_count",
                "rolling_avg_rtt", "rolling_p95_rtt", "rolling_std_rtt",
                # --- Labels ---
                "target_next_avg_rtt"
            ])

    def _write_row(self, metrics, target):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics["timestamp"],
                metrics["avg_rtt_us"],
                metrics["p95_rtt_us"],
                metrics["min_rtt_us"],
                metrics["max_rtt_us"],
                metrics["avg_cwnd"],
                metrics["throughput_bps"],
                metrics["retrans_count"],
                metrics["rolling_avg_rtt"],
                metrics["rolling_p95_rtt"],
                metrics["rolling_std_rtt"],
                target # The Label
            ])

    def _print_row(self, m, target):
        print(
            f"[{m['timestamp']}] "
            f"RTT: {m['avg_rtt_us']:<5} | "
            f"CWND: {m['avg_cwnd']:<4} | "
            f"Tput: {m['throughput_bps']/1024:.1f} KB/s | "
            f"Target(NextRTT): {target}"
        )

def parse_args():
    parser = argparse.ArgumentParser(description="Smart network telemetry collector v2")
    parser.add_argument("--interval", type=float, default=0.05, help="aggregation interval in seconds")
    parser.add_argument("--window", type=int, default=10, help="rolling window length")
    parser.add_argument("--csv", type=str, default="train_data.csv", help="output CSV path")
    parser.add_argument("--max-samples", type=int, default=10000, help="max samples per interval")
    return parser.parse_args()

if __name__ == "__main__":
    main_args = parse_args()
    agent = SmartAgent(main_args.interval, main_args.window, main_args.csv, main_args.max_samples)
    agent.run()