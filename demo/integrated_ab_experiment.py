#!/usr/bin/env python3
"""
Integrated A/B Experiment with Real vLLM

将模拟网络与实际 vLLM 结合的 A/B 实验：

1. 使用 multi_user_hint_server.py 模拟多用户网络
2. 实际调用 vLLM 生成 token
3. 根据模拟的 RTT 计算"有效接收"的 token 数
4. 对比 Baseline vs Network-Aware 的 ETPS

实验流程：
1. 启动 multi_user_hint_server.py
2. 启动 vLLM (设置 VLLM_HINT_SERVER_URL)
3. 运行本脚本进行 A/B 实验
"""

import argparse
import json
import sys
import time
import threading
import requests
from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import deque

import numpy as np


@dataclass
class UserSession:
    """用户会话"""
    user_id: int
    prompt: str
    tokens_generated: int = 0
    tokens_effective: int = 0
    tokens_wasted: int = 0
    start_time: float = 0
    end_time: float = 0
    avg_rtt: float = 0
    avg_health: float = 0


@dataclass
class ExperimentResult:
    """实验结果"""
    mode: str
    total_tokens_generated: int
    total_tokens_effective: int
    total_tokens_wasted: int
    duration: float
    etps: float
    user_sessions: List[UserSession]


class IntegratedExperiment:
    """集成实验"""
    
    def __init__(
        self,
        vllm_url: str = "http://localhost:8000/v1",
        hint_server_url: str = "http://localhost:5000",
        total_budget: float = 100  # tokens/s
    ):
        self.vllm_url = vllm_url
        self.hint_server_url = hint_server_url
        self.total_budget = total_budget
        
        # 模型名称（自动检测）
        self.model_name = self._detect_model()
    
    def _detect_model(self) -> str:
        """检测 vLLM 模型名称"""
        try:
            resp = requests.get(f"{self.vllm_url}/models", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("data"):
                    return data["data"][0]["id"]
        except:
            pass
        return "default"
    
    def _set_mode(self, mode: str):
        """设置 Hint Server 模式"""
        try:
            resp = requests.post(f"{self.hint_server_url}/mode/{mode}", timeout=2)
            if resp.status_code == 200:
                print(f"✅ Mode set to: {mode}")
                return True
        except Exception as e:
            print(f"⚠️ Failed to set mode: {e}")
        return False
    
    def _get_user_allocation(self, user_id: int) -> Dict:
        """获取用户的当前分配信息"""
        try:
            resp = requests.get(f"{self.hint_server_url}/hint?user_id={user_id}", timeout=0.5)
            if resp.status_code == 200:
                return resp.json()
        except:
            pass
        return {"health": 1.0, "metrics": {"max_receive_rate": 50, "rtt": 100}}
    
    def _generate_tokens(
        self,
        prompt: str,
        max_tokens: int = 100,
        user_id: int = 1
    ) -> UserSession:
        """
        生成 token 并计算有效接收数。
        
        核心逻辑：
        - vLLM 按 health_factor 控制生成速率
        - 我们根据模拟的 RTT 计算"有效接收"数
        """
        session = UserSession(
            user_id=user_id,
            prompt=prompt[:50] + "..." if len(prompt) > 50 else prompt,
            start_time=time.time()
        )
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": True
        }
        
        rtt_samples = []
        health_samples = []
        
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
                                session.tokens_generated += 1
                                
                                # 获取当前用户的网络状态
                                alloc = self._get_user_allocation(user_id)
                                rtt = alloc.get("metrics", {}).get("rtt", 100)
                                health = alloc.get("health", 1.0)
                                max_rate = alloc.get("metrics", {}).get("max_receive_rate", 50)
                                
                                rtt_samples.append(rtt)
                                health_samples.append(health)
                                
                                # --- [发送速率 vs 网络容量 模型] ---
                                # 物理原理：
                                # 1. 网络容量 = max_receive_rate (tokens/s)，由 RTT 决定
                                # 2. 发送速率 = 当前 token 生成速率
                                # 3. 如果 发送 > 容量，缓冲区溢出，丢包
                                # 4. 如果 发送 ≤ 容量，几乎不丢包
                                #
                                # max_receive_rate 计算：RTT 低 → 容量大
                                # capacity = 500 / max(rtt, 10) tokens/s
                                # RTT=20ms  → 25 tokens/s
                                # RTT=100ms → 5 tokens/s
                                # RTT=400ms → 1.25 tokens/s
                                
                                network_capacity = 500.0 / max(rtt, 10)  # tokens/s
                                
                                # 计算当前发送速率
                                elapsed = time.time() - session.start_time
                                if elapsed > 0.1:
                                    send_rate = session.tokens_generated / elapsed
                                else:
                                    send_rate = 50  # 默认估计
                                
                                # 丢包率 = max(0, (发送 - 容量) / 发送)
                                if send_rate <= network_capacity:
                                    # 网络能承受，不丢包
                                    session.tokens_effective += 1
                                else:
                                    # 网络过载，按比例丢包
                                    keep_rate = network_capacity / send_rate
                                    if not hasattr(session, '_effective_accumulator'):
                                        session._effective_accumulator = 0.0
                                    session._effective_accumulator += keep_rate
                                    if session._effective_accumulator >= 1.0:
                                        session.tokens_effective += 1
                                        session._effective_accumulator -= 1.0
                                    else:
                                        session.tokens_wasted += 1
                                # --- [END 发送速率模型] ---
                                
                                # 打印进度
                                sys.stdout.write(content)
                                sys.stdout.flush()
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        session.end_time = time.time()
        session.avg_rtt = np.mean(rtt_samples) if rtt_samples else 0
        session.avg_health = np.mean(health_samples) if health_samples else 1.0
        
        return session
    
    def run_group(
        self,
        mode: str,
        prompts: List[str],
        max_tokens: int = 100,
        user_ids: List[int] = [1, 2]
    ) -> ExperimentResult:
        """运行一组实验"""
        print(f"\n{'='*60}")
        print(f"📊 Running {mode.upper()} Group (CONCURRENT)")
        print(f"{'='*60}")
        
        # 设置模式
        self._set_mode(mode)
        time.sleep(0.5)  # 等待模式切换生效
        
        sessions = []
        results_lock = threading.Lock()
        start_time = time.time()
        
        def run_user_session(prompt, user_id):
            """在独立线程中运行用户会话"""
            print(f"\n[User {user_id}] Starting: {prompt[:50]}...")
            session = self._generate_tokens(prompt, max_tokens, user_id)
            with results_lock:
                sessions.append(session)
            print(f"\n✅ User {user_id}: Generated={session.tokens_generated}, "
                  f"Effective={session.tokens_effective}, "
                  f"Wasted={session.tokens_wasted}")
        
        # 并发运行所有用户的请求
        threads = []
        for prompt, user_id in zip(prompts, user_ids):
            t = threading.Thread(target=run_user_session, args=(prompt, user_id))
            threads.append(t)
            t.start()
        
        # 等待所有线程完成
        for t in threads:
            t.join()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 汇总结果
        total_generated = sum(s.tokens_generated for s in sessions)
        total_effective = sum(s.tokens_effective for s in sessions)
        total_wasted = sum(s.tokens_wasted for s in sessions)
        etps = total_effective / duration if duration > 0 else 0
        
        return ExperimentResult(
            mode=mode,
            total_tokens_generated=total_generated,
            total_tokens_effective=total_effective,
            total_tokens_wasted=total_wasted,
            duration=duration,
            etps=etps,
            user_sessions=sessions
        )
    
    def run_experiment(
        self,
        prompts: List[str] = None,
        max_tokens: int = 200,  # 增加 token 数，让请求有更长的并发时间
        user_ids: List[int] = [1, 2]
    ):
        """运行完整 A/B 实验"""
        if prompts is None:
            # 使用更长的 prompts，确保两个请求有足够的重叠时间
            prompts = [
                "Write a very detailed and comprehensive explanation of how neural networks learn through backpropagation, including the mathematical foundations and practical applications.",
                "Explain in great detail the concept of gradient descent and all its variants in machine learning, with examples and comparisons."
            ]
        
        print("\n" + "🚀" * 20)
        print("    INTEGRATED A/B EXPERIMENT")
        print("🚀" * 20)
        print(f"\n📊 Configuration:")
        print(f"   vLLM URL: {self.vllm_url}")
        print(f"   Hint Server: {self.hint_server_url}")
        print(f"   Model: {self.model_name}")
        print(f"   Max Tokens: {max_tokens}")
        print(f"   Users: {user_ids}")
        
        # 运行 Baseline 组
        baseline_result = self.run_group(
            mode="baseline",
            prompts=prompts,
            max_tokens=max_tokens,
            user_ids=user_ids
        )
        
        time.sleep(2)  # 组间间隔
        
        # 运行 Network-Aware 组
        network_aware_result = self.run_group(
            mode="network_aware",
            prompts=prompts,
            max_tokens=max_tokens,
            user_ids=user_ids
        )
        
        # 打印对比结果
        self._print_comparison(baseline_result, network_aware_result)
        
        return baseline_result, network_aware_result
    
    def _print_comparison(self, baseline: ExperimentResult, network_aware: ExperimentResult):
        """打印对比结果"""
        print("\n" + "=" * 60)
        print("📊 EXPERIMENT RESULTS COMPARISON")
        print("=" * 60)
        
        print(f"\n🔴 BASELINE (Equal Allocation):")
        print(f"   Total Generated:  {baseline.total_tokens_generated}")
        print(f"   Total Effective:  {baseline.total_tokens_effective}")
        print(f"   Total Wasted:     {baseline.total_tokens_wasted} "
              f"({baseline.total_tokens_wasted/max(1,baseline.total_tokens_generated)*100:.1f}%)")
        print(f"   Duration:         {baseline.duration:.2f}s")
        print(f"   ETPS:             {baseline.etps:.2f}")
        for s in baseline.user_sessions:
            print(f"   └─ User {s.user_id}: gen={s.tokens_generated}, eff={s.tokens_effective}, "
                  f"rtt={s.avg_rtt:.0f}ms")
        
        print(f"\n🟢 NETWORK-AWARE (Health-Based Allocation):")
        print(f"   Total Generated:  {network_aware.total_tokens_generated}")
        print(f"   Total Effective:  {network_aware.total_tokens_effective}")
        print(f"   Total Wasted:     {network_aware.total_tokens_wasted} "
              f"({network_aware.total_tokens_wasted/max(1,network_aware.total_tokens_generated)*100:.1f}%)")
        print(f"   Duration:         {network_aware.duration:.2f}s")
        print(f"   ETPS:             {network_aware.etps:.2f}")
        for s in network_aware.user_sessions:
            print(f"   └─ User {s.user_id}: gen={s.tokens_generated}, eff={s.tokens_effective}, "
                  f"rtt={s.avg_rtt:.0f}ms, health={s.avg_health:.2f}")
        
        # 计算提升
        if baseline.etps > 0:
            improvement = (network_aware.etps - baseline.etps) / baseline.etps * 100
        else:
            improvement = 0
        
        waste_baseline = baseline.total_tokens_wasted
        waste_network = network_aware.total_tokens_wasted
        waste_reduction = (waste_baseline - waste_network) / max(1, waste_baseline) * 100
        
        print(f"\n📈 IMPROVEMENT:")
        print(f"   ETPS Improvement:   {improvement:+.2f}%")
        print(f"   Waste Reduction:    {waste_reduction:+.2f}%")
        
        if improvement > 0:
            print(f"\n   ✅ Network-Aware scheduling outperforms Baseline!")
        else:
            print(f"\n   ⚠️ Results inconclusive")


def main():
    parser = argparse.ArgumentParser(description="Integrated A/B Experiment")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--hint-url", default="http://localhost:5000")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--num-users", type=int, default=8)  # 增加到 8 个用户！
    args = parser.parse_args()
    
    # 生成多个用户的 prompts
    base_prompts = [
        "Write a detailed explanation of deep learning architectures.",
        "Explain optimization algorithms in machine learning.",
        "Describe the process of training large language models.",
        "Write about the history of artificial intelligence.",
        "Explain how neural networks learn patterns from data.",
        "Describe the transformer architecture in detail.",
        "Write about reinforcement learning algorithms.",
        "Explain the concept of attention mechanisms in AI.",
    ]
    
    # 用户 ID 分配：1-4 网络差；5-8 网络好（需要更新 hint server）
    user_ids = list(range(1, args.num_users + 1))
    prompts = base_prompts[:args.num_users]
    
    experiment = IntegratedExperiment(
        vllm_url=args.vllm_url,
        hint_server_url=args.hint_url
    )
    
    experiment.run_experiment(
        prompts=prompts,
        max_tokens=args.max_tokens,
        user_ids=user_ids
    )


if __name__ == "__main__":
    main()

