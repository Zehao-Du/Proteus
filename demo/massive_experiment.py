#!/usr/bin/env python3
"""
🚀 MASSIVE A/B EXPERIMENT - 8192 Users with TIME LIMIT!

模拟 8192 个用户同时请求 vLLM，验证网络感知调度的效果。

核心机制：
- 设置全局时间限制（如 30 秒）
- 高优先级用户先完成 → 成功
- 低优先级用户可能超时 → 失败
- Network-aware 调度让高健康度用户优先获得 GPU

用户网络分布：
- 20% 很差 (RTT 400-500ms, health ~0.1)
- 30% 差 (RTT 200-400ms, health ~0.3)
- 30% 好 (RTT 50-200ms, health ~0.6)
- 20% 很好 (RTT 10-50ms, health ~0.9)
"""

import argparse
import asyncio
import aiohttp
import json
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict
from collections import defaultdict
import sys


@dataclass
class UserProfile:
    """用户配置"""
    user_id: int
    rtt: float  # ms
    health: float  # 0.0-1.0
    category: str  # 'very_bad', 'bad', 'good', 'very_good'
    patience: float = 0  # 用户耐心值（秒），超时就放弃


@dataclass
class RequestResult:
    """请求结果"""
    user_id: int
    tokens_generated: int = 0
    tokens_effective: int = 0
    start_time: float = 0
    end_time: float = 0
    rtt: float = 0
    health: float = 0
    completed: bool = False  # 是否完成
    timeout: bool = False    # 是否超时


def generate_user_profiles(num_users: int = 8192, time_limit: float = 30.0) -> List[UserProfile]:
    """生成用户配置 - 使用正态分布（有长尾）
    
    RTT 分布：
    - 均值 400ms，标准差 1000ms
    - 截断到 [0, 800000] ms 范围（模拟极端情况：有人 RTT 很小，有人很大）
    - 形成长尾分布：大多数用户网络正常，少数用户网络很差
    
    健康度计算：
    - health = exp(-RTT / 500)
    - RTT 越高，健康度越低
    - 使用 500 而不是 150，以适配新的 RTT 分布 (loc=400, scale=1000)
    
    耐心值：
    - patience = time_limit × (0.3 + 0.7 × health)
    - 网络好的用户愿意等更久
    """
    profiles = []
    
    # 使用正态分布生成 RTT（均值 400ms，标准差 1000ms）
    rtts = np.random.normal(loc=400, scale=1000, size=num_users)
    
    # 截断到合理范围 [0, 800000] ms（模拟极端情况）
    rtts = np.clip(rtts, 0, 800000)
    
    for user_id, rtt in enumerate(rtts, start=1):
        # 根据 RTT 计算健康度：health = exp(-RTT / 500)
        # 使用 500 而不是 150，以适配新的 RTT 分布 (loc=400, scale=1000)
        health = np.exp(-rtt / 500.0)
        
        # 根据 RTT 分类
        if rtt >= 400:
            category = 'very_bad'
        elif rtt >= 200:
            category = 'bad'
        elif rtt >= 80:
            category = 'good'
        else:
            category = 'very_good'
        
        # 耐心值：网络好的用户愿意等更久
        patience = time_limit * (0.3 + 0.7 * health)
        
        profiles.append(UserProfile(
            user_id=user_id,
            rtt=rtt,
            health=health,
            category=category,
            patience=patience
        ))
    
    # 打乱顺序
    np.random.shuffle(profiles)
    return profiles


class MassiveExperiment:
    """大规模实验 - 带时间限制的竞争"""
    
    def __init__(
        self,
        vllm_url: str = "http://localhost:8000/v1",
        num_users: int = 8192,
        max_tokens: int = 50,
        concurrency: int = 256,  # 同时发送的请求数
        time_limit: float = 30.0,  # 全局时间限制（秒）
    ):
        self.vllm_url = vllm_url
        self.num_users = num_users
        self.max_tokens = max_tokens
        self.concurrency = concurrency
        self.time_limit = time_limit
        self.model_name = None
        self.experiment_start_time = 0  # 实验开始时间
        
        # 生成用户配置
        self.user_profiles = generate_user_profiles(num_users, time_limit)
        
        # 统计
        self.results: List[RequestResult] = []
        
    async def detect_model(self, session: aiohttp.ClientSession) -> str:
        """检测模型名称"""
        try:
            async with session.get(f"{self.vllm_url}/models") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = data.get("data", [])
                    if models:
                        return models[0].get("id", "unknown")
        except:
            pass
        return "unknown"
    
    async def send_request(
        self,
        session: aiohttp.ClientSession,
        profile: UserProfile,
        semaphore: asyncio.Semaphore,
        mode: str,
    ) -> RequestResult:
        """发送单个请求 - 带用户耐心超时"""
        result = RequestResult(
            user_id=profile.user_id,
            rtt=profile.rtt,
            health=profile.health,
            start_time=time.time(),
        )
        
        prompt = f"User {profile.user_id}: Write a brief story about AI."
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        
        # 用户耐心超时：网络差的用户更容易放弃
        user_timeout = profile.patience
        
        async with semaphore:
            try:
                async with session.post(
                    f"{self.vllm_url}/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=user_timeout),
                ) as resp:
                    if resp.status != 200:
                        return result
                    
                    effective_accumulator = 0.0
                    first_token_time = None
                    
                    async for line in resp.content:
                        # 检查全局时间限制
                        if time.time() - self.experiment_start_time > self.time_limit:
                            result.timeout = True
                            break
                        
                        if not line:
                            continue
                        
                        line_str = line.decode('utf-8').strip()
                        if not line_str.startswith("data: "):
                            continue
                        
                        data_str = line_str[6:]
                        if data_str == "[DONE]":
                            result.completed = True
                            break
                        
                        try:
                            data = json.loads(data_str)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                if delta.get("content"):
                                    if first_token_time is None:
                                        first_token_time = time.time()
                                    
                                    result.tokens_generated += 1
                                    
                                    # 计算有效率（基于网络容量模型）
                                    elapsed = time.time() - result.start_time
                                    if elapsed > 0.1:
                                        send_rate = result.tokens_generated / elapsed
                                        network_capacity = 500.0 / max(profile.rtt, 10)
                                        
                                        if send_rate <= network_capacity:
                                            result.tokens_effective += 1
                                        else:
                                            keep_rate = network_capacity / send_rate
                                            effective_accumulator += keep_rate
                                            if effective_accumulator >= 1.0:
                                                result.tokens_effective += 1
                                                effective_accumulator -= 1.0
                                    else:
                                        result.tokens_effective += 1
                        except json.JSONDecodeError:
                            continue
                            
            except asyncio.TimeoutError:
                result.timeout = True
            except Exception as e:
                pass
        
        result.end_time = time.time()
        return result
    
    async def run_experiment(self, mode: str) -> Dict:
        """运行实验 - 带时间限制的竞争"""
        print(f"\n{'='*60}")
        print(f"🚀 Running {mode.upper()} - {self.num_users} Users")
        print(f"{'='*60}")
        
        connector = aiohttp.TCPConnector(limit=self.concurrency)
        async with aiohttp.ClientSession(connector=connector) as session:
            # 检测模型
            if not self.model_name:
                self.model_name = await self.detect_model(session)
                print(f"📦 Model: {self.model_name}")
            
            print(f"🔢 Users: {self.num_users}")
            print(f"🎯 Concurrency: {self.concurrency}")
            print(f"📝 Max tokens: {self.max_tokens}")
            print(f"⏱️  Time limit: {self.time_limit}s")
            
            # 设置模式
            try:
                async with session.post(f"http://localhost:5000/mode/{mode}") as resp:
                    if resp.status == 200:
                        print(f"✅ Mode set to: {mode}")
            except:
                print(f"⚠️ Could not set mode (hint server may not be running)")
            
            # 创建信号量控制并发
            semaphore = asyncio.Semaphore(self.concurrency)
            
            # 设置实验开始时间（全局时间限制的起点）
            self.experiment_start_time = time.time()
            start_time = self.experiment_start_time
            
            tasks = [
                self.send_request(session, profile, semaphore, mode)
                for profile in self.user_profiles
            ]
            
            # 使用进度显示
            print(f"\n⏳ Sending {len(tasks)} requests (time limit: {self.time_limit}s)...")
            
            # 使用 asyncio.wait 而不是 gather，这样可以设置超时
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.time_limit + 5  # 额外5秒用于清理
                )
            except asyncio.TimeoutError:
                print(f"⚠️ Global timeout reached!")
                results = []
            
            end_time = time.time()
            duration = min(end_time - start_time, self.time_limit)
        
        # 过滤有效结果
        valid_results = [r for r in results if isinstance(r, RequestResult)]
        
        # 统计结果
        total_generated = sum(r.tokens_generated for r in valid_results)
        total_effective = sum(r.tokens_effective for r in valid_results)
        total_wasted = total_generated - total_effective
        total_completed = sum(1 for r in valid_results if r.completed)
        total_timeout = sum(1 for r in valid_results if r.timeout)
        
        # 按类别统计
        category_stats = defaultdict(lambda: {"gen": 0, "eff": 0, "count": 0, "completed": 0, "timeout": 0})
        for result, profile in zip(valid_results, self.user_profiles):
            cat = profile.category
            category_stats[cat]["gen"] += result.tokens_generated
            category_stats[cat]["eff"] += result.tokens_effective
            category_stats[cat]["count"] += 1
            if result.completed:
                category_stats[cat]["completed"] += 1
            if result.timeout:
                category_stats[cat]["timeout"] += 1
        
        stats = {
            "mode": mode,
            "num_users": self.num_users,
            "duration": duration,
            "total_generated": total_generated,
            "total_effective": total_effective,
            "total_wasted": total_wasted,
            "waste_rate": total_wasted / max(total_generated, 1) * 100,
            "etps": total_effective / duration if duration > 0 else 0,
            "throughput": total_generated / duration if duration > 0 else 0,
            "total_completed": total_completed,
            "total_timeout": total_timeout,
            "completion_rate": total_completed / max(len(valid_results), 1) * 100,
            "category_stats": dict(category_stats),
        }
        
        # 打印结果
        print(f"\n📊 Results:")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Total Generated: {total_generated}")
        print(f"   Total Effective: {total_effective}")
        print(f"   Waste Rate: {stats['waste_rate']:.1f}%")
        print(f"   ETPS: {stats['etps']:.2f}")
        print(f"   Throughput: {stats['throughput']:.2f} tokens/s")
        print(f"   Completed: {total_completed}/{len(valid_results)} ({stats['completion_rate']:.1f}%)")
        print(f"   Timeout: {total_timeout}")
        
        print(f"\n   By Category:")
        for cat in ['very_bad', 'bad', 'good', 'very_good']:
            s = category_stats[cat]
            if s["count"] > 0:
                eff_rate = s["eff"] / max(s["gen"], 1) * 100
                comp_rate = s["completed"] / s["count"] * 100
                print(f"   └─ {cat:10s}: users={s['count']:4d}, gen={s['gen']:5d}, eff={s['eff']:5d} ({eff_rate:.1f}%), completed={s['completed']} ({comp_rate:.1f}%)")
        
        return stats
    
    async def run_ab_experiment(self):
        """运行 A/B 实验 - 时间限制竞争"""
        print("\n" + "🚀" * 30)
        print(f"     MASSIVE A/B EXPERIMENT - {self.num_users} USERS")
        print(f"     TIME LIMIT: {self.time_limit} SECONDS")
        print("🚀" * 30)
        
        # 重新生成用户配置（确保两次实验用同样的用户）
        self.user_profiles = generate_user_profiles(self.num_users, self.time_limit)
        
        # Baseline
        baseline = await self.run_experiment("baseline")
        
        await asyncio.sleep(3)
        
        # 重新生成相同的用户配置
        np.random.seed(42)  # 固定种子确保可重复
        self.user_profiles = generate_user_profiles(self.num_users, self.time_limit)
        
        # Network-Aware
        network_aware = await self.run_experiment("network_aware")
        
        # 对比
        print("\n" + "=" * 60)
        print("📊 COMPARISON")
        print("=" * 60)
        
        print(f"\n🔴 BASELINE:")
        print(f"   ETPS: {baseline['etps']:.2f}")
        print(f"   Waste: {baseline['waste_rate']:.1f}%")
        print(f"   Completed: {baseline['total_completed']}/{self.num_users} ({baseline['completion_rate']:.1f}%)")
        
        print(f"\n🟢 NETWORK-AWARE:")
        print(f"   ETPS: {network_aware['etps']:.2f}")
        print(f"   Waste: {network_aware['waste_rate']:.1f}%")
        print(f"   Completed: {network_aware['total_completed']}/{self.num_users} ({network_aware['completion_rate']:.1f}%)")
        
        improvement = (network_aware['etps'] - baseline['etps']) / max(baseline['etps'], 1) * 100
        waste_reduction = baseline['waste_rate'] - network_aware['waste_rate']
        completion_improvement = network_aware['completion_rate'] - baseline['completion_rate']
        
        print(f"\n📈 IMPROVEMENT:")
        print(f"   ETPS: {improvement:+.2f}%")
        print(f"   Waste Reduction: {waste_reduction:+.2f}%")
        print(f"   Completion Rate: {completion_improvement:+.2f}%")
        
        # 按类别对比
        print(f"\n📊 BY CATEGORY COMPARISON:")
        for cat in ['very_bad', 'bad', 'good', 'very_good']:
            b = baseline['category_stats'].get(cat, {})
            n = network_aware['category_stats'].get(cat, {})
            if b and n:
                b_comp = b.get('completed', 0) / max(b.get('count', 1), 1) * 100
                n_comp = n.get('completed', 0) / max(n.get('count', 1), 1) * 100
                print(f"   └─ {cat:10s}: Baseline {b_comp:.1f}% → Network-Aware {n_comp:.1f}% ({n_comp - b_comp:+.1f}%)")
        
        if improvement > 5:
            print(f"\n   🎉 Network-Aware scheduling wins with {improvement:.1f}% ETPS improvement!")
        elif improvement > 0:
            print(f"\n   ✅ Network-Aware scheduling shows modest improvement")
        else:
            print(f"\n   ⚠️ Results inconclusive - try reducing max-num-seqs")


async def main():
    parser = argparse.ArgumentParser(description="Massive A/B Experiment with Time Limit")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--num-users", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--time-limit", type=float, default=30.0, 
                        help="Time limit in seconds for each experiment run")
    args = parser.parse_args()
    
    experiment = MassiveExperiment(
        vllm_url=args.vllm_url,
        num_users=args.num_users,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        time_limit=args.time_limit,
    )
    
    await experiment.run_ab_experiment()


if __name__ == "__main__":
    np.random.seed(42)  # 确保可重复性
    asyncio.run(main())

