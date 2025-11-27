#!/usr/bin/env python3
"""Smart Agent: eBPF-based TCP RTT & retransmission monitor with richer metrics.

- Collects RTT samples and retransmission events via BCC eBPF probes.
- Aggregates metrics per interval and maintains a rolling window for smoother trends.
- Writes extended metrics to CSV for model training or live visualization.
"""
import argparse
import csv
import statistics
import time
from collections import deque

from bcc import BPF


BPF_PROGRAM = r"""
#include <uapi/linux/ptrace.h>
#include <net/sock.h>
#include <bcc/proto.h>
#include <linux/tcp.h>

BPF_PERF_OUTPUT(rtt_events);
BPF_PERF_OUTPUT(retrans_events);

struct rtt_data_t {
    u32 rtt;
};

struct retrans_data_t {
    u32 dummy;
};

int trace_tcp_rcv(struct pt_regs *ctx, struct sock *sk)
{
    struct tcp_sock *ts = (struct tcp_sock *)sk;
    u32 srtt = ts->srtt_us >> 3;

    if (srtt == 0) return 0;

    struct rtt_data_t data = {};
    data.rtt = srtt;
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

        self.bpf = BPF(text=BPF_PROGRAM)
        self.bpf.attach_kprobe(event="tcp_rcv_established", fn_name="trace_tcp_rcv")
        self.bpf.attach_kprobe(event="tcp_retransmit_skb", fn_name="trace_retransmit")

        self.rtt_samples = []
        self.retrans_count = 0
        self.window_buffer = deque(maxlen=self.window)

        self.bpf["rtt_events"].open_perf_buffer(self._handle_rtt)
        self.bpf["retrans_events"].open_perf_buffer(self._handle_retrans)

    # eBPF callbacks
    def _handle_rtt(self, cpu, data, size):
        event = self.bpf["rtt_events"].event(data)
        if len(self.rtt_samples) < self.max_samples:
            self.rtt_samples.append(event.rtt)

    def _handle_retrans(self, cpu, data, size):
        self.retrans_count += 1

    # Metrics helpers
    def _percentile(self, values, pct):
        if not values:
            return 0
        k = (len(values) - 1) * pct
        f = int(k)
        c = min(f + 1, len(values) - 1)
        if f == c:
            return values[f]
        return values[f] + (values[c] - values[f]) * (k - f)

    def _aggregate_metrics(self):
        if not self.rtt_samples:
            avg_rtt = p95_rtt = min_rtt = max_rtt = 0
        else:
            sorted_rtt = sorted(self.rtt_samples)
            avg_rtt = int(statistics.fmean(sorted_rtt))
            p95_rtt = int(self._percentile(sorted_rtt, 0.95))
            min_rtt = sorted_rtt[0]
            max_rtt = sorted_rtt[-1]

        metrics = {
            "timestamp": int(time.time()),
            "avg_rtt_us": avg_rtt,
            "p95_rtt_us": p95_rtt,
            "min_rtt_us": min_rtt,
            "max_rtt_us": max_rtt,
            "retrans_count": self.retrans_count,
            "rtt_samples": len(self.rtt_samples),
        }

        # update rolling window
        self.window_buffer.append(metrics)
        if self.window_buffer:
            rolling_avg = statistics.fmean(m["avg_rtt_us"] for m in self.window_buffer)
            rolling_p95 = statistics.fmean(m["p95_rtt_us"] for m in self.window_buffer)
        else:
            rolling_avg = rolling_p95 = 0

        metrics.update({
            "rolling_avg_rtt_us": int(rolling_avg),
            "rolling_p95_rtt_us": int(rolling_p95),
        })
        return metrics

    def _reset_counters(self):
        self.rtt_samples = []
        self.retrans_count = 0

    def run(self):
        print("Initializing Smart Agent (eBPF collector with enhanced metrics)...")
        print("Press Ctrl+C to stop. Writing to", self.csv_path)
        self._prepare_csv()
        try:
            while True:
                self._poll_events()
                metrics = self._aggregate_metrics()
                self._write_row(metrics)
                self._print_row(metrics)
                self._reset_counters()
        except KeyboardInterrupt:
            print("\nStopping collector.")

    def _poll_events(self):
        start = time.time()
        while time.time() - start < self.interval:
            timeout_ms = int(max((self.interval - (time.time() - start)) * 1000, 1))
            self.bpf.perf_buffer_poll(timeout=timeout_ms)

    def _prepare_csv(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "avg_rtt_us",
                "p95_rtt_us",
                "min_rtt_us",
                "max_rtt_us",
                "retrans_count",
                "rtt_samples",
                "rolling_avg_rtt_us",
                "rolling_p95_rtt_us",
                "label",
            ])

    def _write_row(self, metrics):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics["timestamp"],
                metrics["avg_rtt_us"],
                metrics["p95_rtt_us"],
                metrics["min_rtt_us"],
                metrics["max_rtt_us"],
                metrics["retrans_count"],
                metrics["rtt_samples"],
                metrics["rolling_avg_rtt_us"],
                metrics["rolling_p95_rtt_us"],
                0,
            ])

    def _print_row(self, metrics):
        print(
            f"{metrics['timestamp']:<15} | avg {metrics['avg_rtt_us']:<8}us | "
            f"p95 {metrics['p95_rtt_us']:<8}us | retrans {metrics['retrans_count']:<5} | "
            f"samples {metrics['rtt_samples']:<5} | roll_avg {metrics['rolling_avg_rtt_us']}us"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Smart network telemetry collector")
    parser.add_argument("--interval", type=float, default=1.0, help="aggregation interval in seconds")
    parser.add_argument("--window", type=int, default=30, help="rolling window length (in intervals)")
    parser.add_argument(
        "--csv",
        type=str,
        default="net_data.csv",
        help="path to write CSV measurements",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="cap RTT samples per interval to avoid unbounded memory use",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    agent = SmartAgent(args.interval, args.window, args.csv, args.max_samples)
    agent.run()


if __name__ == "__main__":
    main()
