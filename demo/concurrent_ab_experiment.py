#!/usr/bin/env python3
"""
Concurrent A/B Experiment - Per-User Network-Aware Scheduling

真正的并发多用户实验：
- 同时向 vLLM 发送多个用户请求
- 每个用户有独立的健康度（通过 Hint Server）
- vLLM 按各用户的健康度分配算力

实验设计：
- 用户1: 网络差 (高 RTT, 低健康度) -> 分配较少算力
- 用户2: 网络好 (低 RTT, 高健康度) -> 分配较多算力

核心指标：
- ETPS (Effective Tokens Per Second): 有效 token/秒
- 浪费率: (生成的 token - 能接收的 token) / 生成的 token
"""

import asyncio
import time
import uuid
import argparse
import json
import csv
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import aiohttp


@dataclass
class UserSession:
    """单用户会话结果"""
    user_id: int
    request_id: str
    mode: str
    prompt_tokens: int
    tokens_generated: int
    tokens_effective: int  # 有效 token = min(生成的, 网络能接收的)
    tokens_wasted: int     # 浪费的 token
    ttft: float  # Time to First Token
    total_time: float
    generation_tps: float  # 生成速率
    effective_tps: float   # 有效速率
    health_factor: float
    network_rtt: float
    network_capacity: float  # 网络能承受的 token/s


class ConcurrentABExperiment:
    """并发 A/B 实验"""
    
    def __init__(self, vllm_url: str, hint_url: str):
        self.vllm_url = vllm_url
        self.hint_url = hint_url
        self.results: list[UserSession] = []
    
    async def set_mode(self, mode: str) -> bool:
        """设置 Hint Server 模式"""
        endpoint = "/mode/baseline" if mode == "baseline" else "/mode/network_aware"
        url = f"{self.hint_url.rsplit('/', 1)[0]}{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        print(f"✅ Mode set to: {mode}")
                        return True
            except Exception as e:
                print(f"⚠️ Failed to set mode: {e}")
        return False
    
    async def get_user_health(self, user_id: int) -> tuple[float, float, float]:
        """获取用户健康度、RTT 和网络容量"""
        url = f"{self.hint_url}?user_id={user_id}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        health = data.get("health", 1.0)
                        metrics = data.get("metrics", {})
                        rtt = metrics.get("rtt", 50)
                        max_receive_rate = metrics.get("max_receive_rate", 100)
                        return health, rtt, max_receive_rate
            except Exception:
                pass
        return 1.0, 50, 100
    
    async def generate_tokens(
        self,
        user_id: int,
        prompt: str,
        max_tokens: int,
        mode: str
    ) -> UserSession:
        """异步生成 token 并测量"""
        request_id = f"user{user_id}_{uuid.uuid4().hex[:8]}"
        
        # 获取健康度和网络能力
        health_factor, rtt, network_capacity = await self.get_user_health(user_id)
        
        start_time = time.time()
        ttft = 0
        tokens_generated = 0
        first_token = True
        
        # 准备请求
        payload = {
            "model": "Qwen/Qwen3-4B-Instruct-2507",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
            "request_id": request_id
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                url = f"{self.vllm_url}/v1/chat/completions"
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status != 200:
                        error = await resp.text()
                        print(f"⚠️ User {user_id} request failed: {error}")
                        return UserSession(
                            user_id=user_id,
                            request_id=request_id,
                            mode=mode,
                            prompt_tokens=len(prompt.split()),
                            tokens_generated=0,
                            tokens_effective=0,
                            tokens_wasted=0,
                            ttft=0,
                            total_time=0,
                            generation_tps=0,
                            effective_tps=0,
                            health_factor=health_factor,
                            network_rtt=rtt,
                            network_capacity=network_capacity
                        )
                    
                    # 处理流式响应
                    async for line in resp.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            data = line[6:]
                            if data == '[DONE]':
                                break
                            try:
                                chunk = json.loads(data)
                                if chunk.get('choices'):
                                    delta = chunk['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        now = time.time()
                                        tokens_generated += 1
                                        
                                        if first_token:
                                            ttft = now - start_time
                                            first_token = False
                            except json.JSONDecodeError:
                                pass
            
            except asyncio.TimeoutError:
                print(f"⚠️ User {user_id} request timeout")
            except Exception as e:
                print(f"⚠️ User {user_id} error: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 计算指标
        generation_tps = tokens_generated / total_time if total_time > 0 else 0
        
        # 有效 token = min(生成的, 网络能在这段时间内接收的)
        # 网络容量是 token/s，总共能接收 capacity * time
        max_receivable = int(network_capacity * total_time)
        tokens_effective = min(tokens_generated, max_receivable)
        tokens_wasted = tokens_generated - tokens_effective
        
        effective_tps = tokens_effective / total_time if total_time > 0 else 0
        
        return UserSession(
            user_id=user_id,
            request_id=request_id,
            mode=mode,
            prompt_tokens=len(prompt.split()),
            tokens_generated=tokens_generated,
            tokens_effective=tokens_effective,
            tokens_wasted=tokens_wasted,
            ttft=ttft,
            total_time=total_time,
            generation_tps=generation_tps,
            effective_tps=effective_tps,
            health_factor=health_factor,
            network_rtt=rtt,
            network_capacity=network_capacity
        )
    
    async def run_concurrent_session(
        self,
        user_ids: list[int],
        prompts: list[str],
        max_tokens: int,
        mode: str
    ) -> list[UserSession]:
        """并发运行多用户会话"""
        print(f"\n🚀 Starting concurrent session: {mode.upper()}")
        print(f"   Users: {user_ids}")
        
        # 设置模式
        await self.set_mode(mode)
        await asyncio.sleep(1)  # 等待模式生效
        
        # 并发发送请求
        tasks = [
            self.generate_tokens(uid, prompt, max_tokens, mode)
            for uid, prompt in zip(user_ids, prompts)
        ]
        
        sessions = await asyncio.gather(*tasks)
        return list(sessions)
    
    async def run_experiment(
        self,
        num_rounds: int = 5,
        max_tokens: int = 100
    ):
        """运行完整实验"""
        prompts = [
            "Explain the concept of machine learning in detail with examples.",
            "What is quantum computing and how does it work in modern computers?"
        ]
        user_ids = [1, 2]  # 用户1网络差，用户2网络好
        
        print("=" * 70)
        print("  Concurrent A/B Experiment - Per-User Network-Aware Scheduling")
        print("=" * 70)
        print(f"\n📋 Configuration:")
        print(f"   - vLLM URL: {self.vllm_url}")
        print(f"   - Hint Server: {self.hint_url}")
        print(f"   - Rounds: {num_rounds}")
        print(f"   - Max Tokens: {max_tokens}")
        print(f"   - Users: {len(user_ids)}")
        print(f"\n📊 User Network Profiles:")
        print(f"   - User 1: Poor network (high RTT ~150ms, low capacity)")
        print(f"   - User 2: Good network (low RTT ~30ms, high capacity)")
        
        all_results = []
        
        for round_num in range(1, num_rounds + 1):
            print(f"\n{'='*70}")
            print(f"  ROUND {round_num}/{num_rounds}")
            print(f"{'='*70}")
            
            # A组: Baseline
            baseline_sessions = await self.run_concurrent_session(
                user_ids, prompts, max_tokens, "baseline"
            )
            all_results.extend(baseline_sessions)
            
            # 打印 Baseline 结果
            print("\n📊 Baseline Results:")
            for s in baseline_sessions:
                waste_pct = s.tokens_wasted / s.tokens_generated * 100 if s.tokens_generated > 0 else 0
                print(f"   User {s.user_id}: Gen={s.tokens_generated}, Eff={s.tokens_effective}, "
                      f"Waste={s.tokens_wasted} ({waste_pct:.0f}%), "
                      f"TPS={s.generation_tps:.1f}, RTT={s.network_rtt:.0f}ms")
            
            await asyncio.sleep(2)  # 间隔
            
            # B组: Network-Aware
            aware_sessions = await self.run_concurrent_session(
                user_ids, prompts, max_tokens, "network_aware"
            )
            all_results.extend(aware_sessions)
            
            # 打印 Network-Aware 结果
            print("\n📊 Network-Aware Results:")
            for s in aware_sessions:
                waste_pct = s.tokens_wasted / s.tokens_generated * 100 if s.tokens_generated > 0 else 0
                print(f"   User {s.user_id}: Gen={s.tokens_generated}, Eff={s.tokens_effective}, "
                      f"Waste={s.tokens_wasted} ({waste_pct:.0f}%), "
                      f"TPS={s.generation_tps:.1f}, Health={s.health_factor:.3f}")
            
            await asyncio.sleep(2)
        
        # 保存结果
        self.results = all_results
        self._save_results()
        self._print_summary()
    
    def _save_results(self):
        """保存结果到 CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"concurrent_ab_results_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=asdict(self.results[0]).keys())
            writer.writeheader()
            for r in self.results:
                writer.writerow(asdict(r))
        
        print(f"\n💾 Results saved to: {filename}")
    
    def _print_summary(self):
        """打印汇总统计"""
        print("\n" + "=" * 70)
        print("  EXPERIMENT SUMMARY")
        print("=" * 70)
        
        # 按模式分组
        baseline = [r for r in self.results if r.mode == "baseline"]
        aware = [r for r in self.results if r.mode == "network_aware"]
        
        def calc_stats(sessions: list[UserSession]):
            if not sessions:
                return {}
            total_generated = sum(s.tokens_generated for s in sessions)
            total_effective = sum(s.tokens_effective for s in sessions)
            total_wasted = sum(s.tokens_wasted for s in sessions)
            avg_ttft = sum(s.ttft for s in sessions) / len(sessions)
            avg_gen_tps = sum(s.generation_tps for s in sessions) / len(sessions)
            avg_eff_tps = sum(s.effective_tps for s in sessions) / len(sessions)
            waste_rate = total_wasted / total_generated * 100 if total_generated > 0 else 0
            return {
                "total_generated": total_generated,
                "total_effective": total_effective,
                "total_wasted": total_wasted,
                "waste_rate": waste_rate,
                "avg_ttft": avg_ttft,
                "avg_gen_tps": avg_gen_tps,
                "avg_eff_tps": avg_eff_tps
            }
        
        baseline_stats = calc_stats(baseline)
        aware_stats = calc_stats(aware)
        
        print("\n📊 Baseline (平均分配):")
        print(f"   Total Generated:  {baseline_stats['total_generated']} tokens")
        print(f"   Total Effective:  {baseline_stats['total_effective']} tokens")
        print(f"   Total Wasted:     {baseline_stats['total_wasted']} tokens")
        print(f"   Waste Rate:       {baseline_stats['waste_rate']:.1f}%")
        print(f"   Avg Gen TPS:      {baseline_stats['avg_gen_tps']:.2f}")
        print(f"   Avg Eff TPS:      {baseline_stats['avg_eff_tps']:.2f}")
        print(f"   Avg TTFT:         {baseline_stats['avg_ttft']:.3f}s")
        
        print("\n📊 Network-Aware (按健康度分配):")
        print(f"   Total Generated:  {aware_stats['total_generated']} tokens")
        print(f"   Total Effective:  {aware_stats['total_effective']} tokens")
        print(f"   Total Wasted:     {aware_stats['total_wasted']} tokens")
        print(f"   Waste Rate:       {aware_stats['waste_rate']:.1f}%")
        print(f"   Avg Gen TPS:      {aware_stats['avg_gen_tps']:.2f}")
        print(f"   Avg Eff TPS:      {aware_stats['avg_eff_tps']:.2f}")
        print(f"   Avg TTFT:         {aware_stats['avg_ttft']:.3f}s")
        
        # 计算改进
        if baseline_stats['avg_eff_tps'] > 0:
            eff_tps_improvement = (aware_stats['avg_eff_tps'] - baseline_stats['avg_eff_tps']) / baseline_stats['avg_eff_tps'] * 100
            waste_reduction = baseline_stats['waste_rate'] - aware_stats['waste_rate']
            
            print("\n" + "=" * 70)
            print("  📈 IMPROVEMENT")
            print("=" * 70)
            print(f"   Effective TPS Improvement: {eff_tps_improvement:+.1f}%")
            print(f"   Waste Rate Reduction:      {waste_reduction:+.1f}%")
            
            if waste_reduction > 0:
                print(f"\n   ✅ Network-aware scheduling reduced waste!")
            else:
                print(f"\n   ⚠️  Results may vary due to network simulation variance")
        
        # 分用户统计
        print("\n" + "=" * 70)
        print("  📊 PER-USER BREAKDOWN")
        print("=" * 70)
        
        for user_id in [1, 2]:
            user_baseline = [r for r in baseline if r.user_id == user_id]
            user_aware = [r for r in aware if r.user_id == user_id]
            
            bl_gen = sum(s.tokens_generated for s in user_baseline)
            bl_eff = sum(s.tokens_effective for s in user_baseline)
            bl_waste = bl_gen - bl_eff
            
            aw_gen = sum(s.tokens_generated for s in user_aware)
            aw_eff = sum(s.tokens_effective for s in user_aware)
            aw_waste = aw_gen - aw_eff
            
            avg_rtt = sum(s.network_rtt for s in user_baseline + user_aware) / len(user_baseline + user_aware) if user_baseline + user_aware else 0
            
            print(f"\n   User {user_id} (avg RTT: {avg_rtt:.0f}ms):")
            print(f"     Baseline:      Gen={bl_gen}, Eff={bl_eff}, Waste={bl_waste}")
            print(f"     Network-Aware: Gen={aw_gen}, Eff={aw_eff}, Waste={aw_waste}")
        
        print("\n" + "=" * 70)


async def main():
    parser = argparse.ArgumentParser(description="Concurrent A/B Experiment")
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--hint-url", default="http://localhost:5000/hint")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()
    
    experiment = ConcurrentABExperiment(args.vllm_url, args.hint_url)
    await experiment.run_experiment(
        num_rounds=args.rounds,
        max_tokens=args.max_tokens
    )


if __name__ == "__main__":
    asyncio.run(main())
