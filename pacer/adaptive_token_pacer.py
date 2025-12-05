#!/usr/bin/env python3
"""Network-aware token pacing for LLM streaming.

This utility ties the Smart Agent's TCP telemetry to token emission rate so that
LLM servers generate tokens at a pace the client link can sustain. The pacer
reads the latest RTT and retransmission metrics from ``net_data.csv`` (or a
custom path), dynamically adjusts target tokens per second, and emits tokens at
that rate to avoid buffering waste during degraded network conditions.

两种使用方式：
1) 本地 CLI：读取 CSV 实时调整输出令牌节奏，演示和本地对接最简单。
2) 云端 Hints Server：以 HTTP JSON 形式对外提供 ``current_tps`` 建议，云端
   LLM 服务可在生成循环中定期查询并动态 ``sleep``，实现按链路质量匀速生成。
"""
import argparse
import json
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _read_latest_metrics(csv_path: Path) -> Optional[Dict[str, float]]:
    """Return the newest metric row from the Smart Agent CSV."""
    if not csv_path.exists():
        return None

    with csv_path.open() as f:
        rows = f.readlines()
    if len(rows) < 2:
        return None

    header = [col.strip() for col in rows[0].strip().split(",")]
    values = [col.strip() for col in rows[-1].strip().split(",")]
    if len(header) != len(values):
        return None

    record: Dict[str, float] = {}
    for key, val in zip(header, values):
        try:
            record[key] = float(val)
        except ValueError:
            record[key] = 0.0
    return record


class AdaptiveTokenPacer:
    """Compute target token rate from live network metrics."""

    def __init__(
        self,
        max_tps: float,
        min_tps: float,
        rtt_baseline_us: float,
        retrans_budget: float,
        smoothing: float,
    ) -> None:
        self.max_tps = max_tps
        self.min_tps = min_tps
        self.rtt_baseline_us = rtt_baseline_us
        self.retrans_budget = retrans_budget
        self.smoothing = smoothing
        self.current_tps = max_tps
        self._lock = threading.Lock()

    def update(self, metrics: Optional[Dict[str, float]]) -> float:
        """Blend a new target rate from RTT and retransmission health."""
        if metrics is None:
            return self.current_tps

        p95_rtt = metrics.get("p95_rtt_us", 0.0)
        retrans = metrics.get("retrans_count", 0.0)

        quality = 1.0
        if p95_rtt > 0:
            rtt_factor = min(1.0, self.rtt_baseline_us / p95_rtt)
            quality *= rtt_factor

        if retrans > self.retrans_budget:
            quality *= self.retrans_budget / max(retrans, 1e-6)

        target = max(self.min_tps, min(self.max_tps, quality * self.max_tps))
        new_tps = (1 - self.smoothing) * self.current_tps + self.smoothing * target

        with self._lock:
            self.current_tps = new_tps
            return self.current_tps

    def get_rate(self) -> float:
        with self._lock:
            return self.current_tps


class PacingHintServer(threading.Thread):
    """轻量级 HTTP 服务，云端侧定期查询得到最新建议的 tokens/s。"""

    def __init__(self, pacer: AdaptiveTokenPacer, metrics_path: Path, refresh_seconds: float, port: int) -> None:
        super().__init__(daemon=True)
        self.pacer = pacer
        self.metrics_path = metrics_path
        self.refresh_seconds = refresh_seconds
        self.port = port
        self._stop_event = threading.Event()

    def run(self) -> None:
        from http.server import BaseHTTPRequestHandler, HTTPServer

        pacer = self.pacer
        metrics_path = self.metrics_path
        refresh_seconds = self.refresh_seconds

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:  # noqa: N802 - keep silent
                return

            def do_GET(self) -> None:  # noqa: N802 - HTTP verb naming
                now = time.time()
                if not hasattr(self.server, "_last_refresh") or now - self.server._last_refresh >= refresh_seconds:
                    metrics = _read_latest_metrics(metrics_path)
                    pacer.update(metrics)
                    self.server._last_refresh = now

                payload = {
                    "current_tps": pacer.get_rate(),
                    "sleep_seconds": 1.0 / max(pacer.get_rate(), 1e-6),
                }
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        httpd = HTTPServer(("0.0.0.0", self.port), Handler)
        while not self._stop_event.is_set():
            httpd.handle_request()

    def stop(self) -> None:
        self._stop_event.set()
        # trigger one dummy request so handle_request returns
        try:
            import socket

            with socket.create_connection(("127.0.0.1", self.port), timeout=0.2) as s:
                s.sendall(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
        except OSError:
            pass


def _tokenize(text: str, mode: str) -> List[str]:
    if mode == "char":
        return list(text)
    return text.split()


def _stream_tokens(
    tokens: Iterable[str],
    pacer: AdaptiveTokenPacer,
    metrics_path: Path,
    refresh_seconds: float,
) -> None:
    last_refresh = 0.0
    for token in tokens:
        now = time.time()
        if now - last_refresh >= refresh_seconds:
            metrics = _read_latest_metrics(metrics_path)
            pacer.update(metrics)
            last_refresh = now

        delay = 1.0 / max(pacer.current_tps, 1e-6)
        print(token, flush=True)
        time.sleep(delay)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive token pacing based on network telemetry")
    parser.add_argument("text", help="sample text to stream as tokens")
    parser.add_argument("--csv", type=Path, default=Path("net_data.csv"), help="Smart Agent CSV path")
    parser.add_argument("--max-tps", type=float, default=40.0, help="upper bound tokens per second")
    parser.add_argument("--min-tps", type=float, default=2.0, help="floor tokens per second")
    parser.add_argument("--rtt-baseline-us", type=float, default=40_000.0, help="healthy p95 RTT (microseconds)")
    parser.add_argument("--retrans-budget", type=float, default=2.0, help="acceptable retransmissions per interval")
    parser.add_argument("--smoothing", type=float, default=0.2, help="EMA smoothing factor for rate changes")
    parser.add_argument("--refresh", type=float, default=1.0, help="seconds between metric reads")
    parser.add_argument("--mode", choices=["word", "char"], default="word", help="tokenization granularity")
    parser.add_argument("--serve", type=int, default=None, help="start HTTP hint server on the given port")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pacer = AdaptiveTokenPacer(
        max_tps=args.max_tps,
        min_tps=args.min_tps,
        rtt_baseline_us=args.rtt_baseline_us,
        retrans_budget=args.retrans_budget,
        smoothing=args.smoothing,
    )

    if args.serve:
        server = PacingHintServer(pacer, args.csv, args.refresh, args.serve)
        server.start()
        print(f"Serving pacing hints on 0.0.0.0:{args.serve}. GET to receive {{'current_tps': ..., 'sleep_seconds': ...}}", flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            server.stop()
    else:
        tokens = _tokenize(args.text, mode=args.mode)
        _stream_tokens(tokens, pacer, args.csv, args.refresh)


if __name__ == "__main__":
    main()
