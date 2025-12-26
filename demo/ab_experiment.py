#!/usr/bin/env python3
"""
A/B Experiment Runner for Network-Aware Token Pacing

This script runs controlled experiments to compare:
- Group A (Pacing ON): vLLM with network-aware scheduling enabled
- Group B (Pacing OFF): vLLM with network-aware scheduling disabled

Key Metric: ETPS (Effective Tokens Per Second)
ETPS = (成功在客户端渲染的 token 数) / 完整会话时间

Usage:
    # Run experiment with 10 sessions per group
    python ab_experiment.py --sessions 10 --prompt "Tell me a story about AI"
    
    # With network chaos injection
    python ab_experiment.py --sessions 5 --enable-chaos --chaos-interval 10
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any

import requests


@dataclass
class SessionResult:
    """Single session result."""
    session_id: int
    group: str  # "pacing_on" or "pacing_off"
    prompt: str
    total_tokens: int
    successful_tokens: int  # Tokens successfully rendered (no errors)
    session_duration: float  # Total time in seconds
    first_token_latency: float  # Time to first token (TTFT)
    etps: float  # Effective Tokens Per Second
    avg_health: float  # Average health factor during session
    avg_rtt: float  # Average RTT during session
    retransmits: int  # Total retransmits observed
    errors: int  # Number of errors/failures
    timestamp: str


class ExperimentRunner:
    """A/B Experiment Runner."""
    
    def __init__(
        self,
        vllm_url: str = "http://localhost:8000/v1",
        hint_url: str = "http://localhost:5000/hint",
        output_dir: str = "ab_results"
    ):
        self.vllm_url = vllm_url
        self.hint_url = hint_url
        self.output_dir = output_dir
        self.results: List[SessionResult] = []
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Health tracking
        self._health_samples = []
        self._rtt_samples = []
        self._retrans_count = 0
        self._monitor_running = False
        
    def _get_model_name(self) -> str:
        """Auto-detect vLLM model name."""
        try:
            resp = requests.get(f"{self.vllm_url}/models", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("data"):
                    return data["data"][0]["id"]
        except:
            pass
        return "default"
    
    def _monitor_network(self):
        """Background thread to sample network health."""
        while self._monitor_running:
            try:
                resp = requests.get(self.hint_url, timeout=0.2)
                if resp.status_code == 200:
                    data = resp.json()
                    self._health_samples.append(data.get("health", 1.0))
                    metrics = data.get("metrics", {})
                    self._rtt_samples.append(metrics.get("rtt", 0))
                    self._retrans_count += metrics.get("retrans", 0)
            except:
                pass
            time.sleep(0.2)
    
    def _start_monitoring(self):
        """Start network monitoring thread."""
        self._health_samples = []
        self._rtt_samples = []
        self._retrans_count = 0
        self._monitor_running = True
        self._monitor_thread = threading.Thread(target=self._monitor_network, daemon=True)
        self._monitor_thread.start()
    
    def _stop_monitoring(self) -> tuple:
        """Stop monitoring and return averages."""
        self._monitor_running = False
        time.sleep(0.3)  # Wait for thread to finish
        
        avg_health = sum(self._health_samples) / len(self._health_samples) if self._health_samples else 1.0
        avg_rtt = sum(self._rtt_samples) / len(self._rtt_samples) if self._rtt_samples else 0
        
        return avg_health, avg_rtt, self._retrans_count
    
    def run_single_session(
        self,
        session_id: int,
        group: str,
        prompt: str,
        max_tokens: int = 200,
        model: str = "default"
    ) -> SessionResult:
        """Run a single LLM session and measure ETPS."""
        
        # Prepare request
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": True
        }
        
        # Start monitoring
        self._start_monitoring()
        
        # Metrics
        start_time = time.time()
        first_token_time = None
        total_tokens = 0
        successful_tokens = 0
        errors = 0
        
        try:
            response = requests.post(
                f"{self.vllm_url}/chat/completions",
                json=payload,
                stream=True,
                timeout=120
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if not line:
                    continue
                
                if line.startswith(b"data: "):
                    data_str = line[6:].decode('utf-8')
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                # First token timing
                                if first_token_time is None:
                                    first_token_time = time.time()
                                
                                total_tokens += 1
                                successful_tokens += 1
                                
                                # Print progress
                                sys.stdout.write(content)
                                sys.stdout.flush()
                    except json.JSONDecodeError:
                        errors += 1
                        
        except requests.exceptions.RequestException as e:
            errors += 1
            print(f"\n❌ Request error: {e}")
        except Exception as e:
            errors += 1
            print(f"\n❌ Unexpected error: {e}")
        
        # Calculate metrics
        end_time = time.time()
        session_duration = end_time - start_time
        first_token_latency = (first_token_time - start_time) if first_token_time else session_duration
        etps = successful_tokens / session_duration if session_duration > 0 else 0
        
        # Stop monitoring
        avg_health, avg_rtt, retrans = self._stop_monitoring()
        
        # Create result
        result = SessionResult(
            session_id=session_id,
            group=group,
            prompt=prompt[:50] + "..." if len(prompt) > 50 else prompt,
            total_tokens=total_tokens,
            successful_tokens=successful_tokens,
            session_duration=round(session_duration, 3),
            first_token_latency=round(first_token_latency, 3),
            etps=round(etps, 2),
            avg_health=round(avg_health, 3),
            avg_rtt=round(avg_rtt, 1),
            retransmits=retrans,
            errors=errors,
            timestamp=datetime.now().isoformat()
        )
        
        print(f"\n✅ Session {session_id} ({group}): ETPS={etps:.2f}, Tokens={successful_tokens}, Duration={session_duration:.2f}s")
        
        return result
    
    def set_pacing_mode(self, enabled: bool) -> bool:
        """
        Set pacing mode via Hint Server API.
        
        Uses the /mode/on and /mode/off endpoints of hint_server_ab.py
        to dynamically switch between network-aware pacing and baseline mode.
        
        Returns:
            True if mode was set successfully, False otherwise.
        """
        mode = "ON" if enabled else "OFF"
        endpoint = "/mode/on" if enabled else "/mode/off"
        
        # Extract base URL from hint_url
        base_url = self.hint_url.rsplit('/hint', 1)[0]
        mode_url = base_url + endpoint
        
        print(f"\n{'='*60}")
        print(f"🔧 Setting Pacing Mode: {mode}")
        
        try:
            resp = requests.post(mode_url, timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                print(f"✅ Mode set successfully: {data.get('message', 'OK')}")
                print(f"{'='*60}\n")
                return True
            else:
                print(f"⚠️ Failed to set mode (status {resp.status_code})")
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Could not reach Hint Server: {e}")
            print(f"   (Continuing with current mode)")
        
        print(f"{'='*60}\n")
        return False
    
    def run_experiment(
        self,
        num_sessions: int,
        prompt: str,
        max_tokens: int = 200,
        warmup_sessions: int = 2
    ):
        """Run complete A/B experiment."""
        
        model = self._get_model_name()
        print(f"🚀 Starting A/B Experiment")
        print(f"   Model: {model}")
        print(f"   Sessions per group: {num_sessions}")
        print(f"   Warmup sessions: {warmup_sessions}")
        print(f"   Max tokens: {max_tokens}")
        print()
        
        # Warmup
        print("🔥 Running warmup sessions...")
        for i in range(warmup_sessions):
            self.run_single_session(
                session_id=-1,
                group="warmup",
                prompt=prompt,
                max_tokens=50,
                model=model
            )
            time.sleep(1)
        
        # Group A: Pacing ON
        print("\n" + "="*60)
        print("📊 GROUP A: Pacing ON (Network-Aware Scheduling)")
        print("="*60)
        self.set_pacing_mode(enabled=True)
        time.sleep(1)  # Let mode change propagate
        
        for i in range(num_sessions):
            result = self.run_single_session(
                session_id=i+1,
                group="pacing_on",
                prompt=prompt,
                max_tokens=max_tokens,
                model=model
            )
            self.results.append(result)
            time.sleep(2)  # Cooldown between sessions
        
        # Group B: Pacing OFF (baseline mode via Hint Server API)
        print("\n" + "="*60)
        print("📊 GROUP B: Pacing OFF (Baseline - Full Speed)")
        print("="*60)
        self.set_pacing_mode(enabled=False)
        time.sleep(1)  # Let mode change propagate
        
        for i in range(num_sessions):
            result = self.run_single_session(
                session_id=i+1,
                group="pacing_off",
                prompt=prompt,
                max_tokens=max_tokens,
                model=model
            )
            self.results.append(result)
            time.sleep(2)
        
        # Restore pacing mode after experiment
        self.set_pacing_mode(enabled=True)
        
        # Save results
        self._save_results()
        self._print_summary()
    
    def _save_results(self):
        """Save results to CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"ab_results_{timestamp}.csv")
        
        with open(filename, 'w', newline='') as f:
            if self.results:
                writer = csv.DictWriter(f, fieldnames=asdict(self.results[0]).keys())
                writer.writeheader()
                for result in self.results:
                    writer.writerow(asdict(result))
        
        # Also save as latest.csv for dashboard
        latest_file = os.path.join(self.output_dir, "latest.csv")
        with open(latest_file, 'w', newline='') as f:
            if self.results:
                writer = csv.DictWriter(f, fieldnames=asdict(self.results[0]).keys())
                writer.writeheader()
                for result in self.results:
                    writer.writerow(asdict(result))
        
        print(f"\n💾 Results saved to: {filename}")
        print(f"💾 Latest results: {latest_file}")
    
    def _print_summary(self):
        """Print experiment summary."""
        pacing_on = [r for r in self.results if r.group == "pacing_on"]
        pacing_off = [r for r in self.results if r.group == "pacing_off"]
        
        def calc_stats(results):
            if not results:
                return {}
            etps_vals = [r.etps for r in results]
            ttft_vals = [r.first_token_latency for r in results]
            return {
                "count": len(results),
                "avg_etps": sum(etps_vals) / len(etps_vals),
                "min_etps": min(etps_vals),
                "max_etps": max(etps_vals),
                "avg_ttft": sum(ttft_vals) / len(ttft_vals),
                "total_errors": sum(r.errors for r in results),
                "total_retrans": sum(r.retransmits for r in results)
            }
        
        stats_on = calc_stats(pacing_on)
        stats_off = calc_stats(pacing_off)
        
        print("\n" + "="*60)
        print("📈 EXPERIMENT SUMMARY")
        print("="*60)
        
        print("\n🟢 Group A (Pacing ON):")
        print(f"   Sessions: {stats_on.get('count', 0)}")
        print(f"   Avg ETPS: {stats_on.get('avg_etps', 0):.2f}")
        print(f"   ETPS Range: {stats_on.get('min_etps', 0):.2f} - {stats_on.get('max_etps', 0):.2f}")
        print(f"   Avg TTFT: {stats_on.get('avg_ttft', 0):.3f}s")
        print(f"   Total Errors: {stats_on.get('total_errors', 0)}")
        print(f"   Total Retransmits: {stats_on.get('total_retrans', 0)}")
        
        print("\n🔴 Group B (Pacing OFF):")
        print(f"   Sessions: {stats_off.get('count', 0)}")
        print(f"   Avg ETPS: {stats_off.get('avg_etps', 0):.2f}")
        print(f"   ETPS Range: {stats_off.get('min_etps', 0):.2f} - {stats_off.get('max_etps', 0):.2f}")
        print(f"   Avg TTFT: {stats_off.get('avg_ttft', 0):.3f}s")
        print(f"   Total Errors: {stats_off.get('total_errors', 0)}")
        print(f"   Total Retransmits: {stats_off.get('total_retrans', 0)}")
        
        # ETPS Improvement
        if stats_on.get('avg_etps') and stats_off.get('avg_etps'):
            improvement = ((stats_on['avg_etps'] - stats_off['avg_etps']) / stats_off['avg_etps']) * 100
            print(f"\n📊 ETPS Improvement: {improvement:+.1f}%")
            if improvement > 0:
                print("   ✅ Pacing ON outperforms baseline!")
            else:
                print("   ⚠️ Baseline performs better (may need network stress)")


class ChaosInjector:
    """Network chaos injection using tc."""
    
    def __init__(self, interface: str = "eth0"):
        self.interface = interface
        self.chaos_active = False
    
    def add_delay(self, delay_ms: int = 100, jitter_ms: int = 50):
        """Add network delay."""
        cmd = f"sudo tc qdisc add dev {self.interface} root netem delay {delay_ms}ms {jitter_ms}ms"
        subprocess.run(cmd.split(), check=False)
        self.chaos_active = True
        print(f"🌪️ Chaos: Added {delay_ms}ms delay with {jitter_ms}ms jitter")
    
    def add_loss(self, loss_pct: float = 5.0):
        """Add packet loss."""
        cmd = f"sudo tc qdisc add dev {self.interface} root netem loss {loss_pct}%"
        subprocess.run(cmd.split(), check=False)
        self.chaos_active = True
        print(f"🌪️ Chaos: Added {loss_pct}% packet loss")
    
    def clear(self):
        """Clear all network chaos rules."""
        cmd = f"sudo tc qdisc del dev {self.interface} root"
        subprocess.run(cmd.split(), check=False, stderr=subprocess.DEVNULL)
        self.chaos_active = False
        print("🧹 Chaos: Cleared all rules")


def main():
    parser = argparse.ArgumentParser(description="A/B Experiment Runner for Token Pacing")
    
    parser.add_argument("--sessions", type=int, default=5,
                       help="Number of sessions per group (default: 5)")
    parser.add_argument("--prompt", type=str, 
                       default="Write a detailed explanation of how neural networks learn.",
                       help="Prompt for LLM generation")
    parser.add_argument("--max-tokens", type=int, default=200,
                       help="Maximum tokens per session (default: 200)")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1",
                       help="vLLM API URL")
    parser.add_argument("--hint-url", type=str, default="http://localhost:5000/hint",
                       help="Hint Server URL")
    parser.add_argument("--output-dir", type=str, default="ab_results",
                       help="Output directory for results")
    
    # Chaos injection
    parser.add_argument("--enable-chaos", action="store_true",
                       help="Enable network chaos injection")
    parser.add_argument("--chaos-delay", type=int, default=100,
                       help="Chaos delay in ms (default: 100)")
    parser.add_argument("--chaos-loss", type=float, default=2.0,
                       help="Chaos packet loss percentage (default: 2.0)")
    parser.add_argument("--chaos-interface", type=str, default="eth0",
                       help="Network interface for chaos (default: eth0)")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = ExperimentRunner(
        vllm_url=args.vllm_url,
        hint_url=args.hint_url,
        output_dir=args.output_dir
    )
    
    # Chaos injection
    chaos = None
    if args.enable_chaos:
        chaos = ChaosInjector(interface=args.chaos_interface)
        print("\n🌪️ Chaos injection enabled!")
        chaos.add_delay(args.chaos_delay, args.chaos_delay // 2)
    
    try:
        # Run experiment
        runner.run_experiment(
            num_sessions=args.sessions,
            prompt=args.prompt,
            max_tokens=args.max_tokens
        )
    finally:
        # Cleanup chaos
        if chaos:
            chaos.clear()
    
    print("\n✅ Experiment complete!")
    print(f"📊 View results with: streamlit run demo/ab_dashboard.py")


if __name__ == "__main__":
    main()

