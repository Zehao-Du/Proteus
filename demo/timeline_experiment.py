#!/usr/bin/env python3
"""
🕐 Timeline Experiment - 验证 Network-Aware 调度的真正优势

核心洞察：
- GPU 生成速度固定，但 chunk 到达客户端的时间不同
- 网络好用户：chunk 立即到达
- 网络差用户：chunk 延迟到达
- Network-aware 优先网络好用户 → 累计有效 chunk 曲线一直在上面

输出：累计有效 chunk 随时间变化的曲线图
"""

import argparse
import asyncio
import aiohttp
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from collections import defaultdict
import sys


@dataclass
class UserProfile:
    """用户配置"""
    user_id: int
    rtt: float  # ms，影响 chunk 到达时间
    health: float  # 0.0-1.0
    category: str  # 'very_bad', 'bad', 'good', 'very_good'


@dataclass
class ChunkEvent:
    """Chunk 事件：记录每个 SSE chunk 的观测时间和合成到达时间
    
    注意：observed_arrival_time 是客户端实际收到 SSE chunk 的时间
    （在 localhost 场景下，近似等于 GPU 生成时间）
    synthetic_arrival_time 是加入网络 RTT 延迟后的"有效到达时间"
    """
    user_id: int
    chunk_idx: int
    observed_arrival_time: float  # 客户端观测到的到达时间（≈ GPU 生成时间，localhost）
    synthetic_arrival_time: float  # 加入 RTT 延迟后的合成到达时间（用于计算有效吞吐）
    rtt: float
    category: str
    chunk_length: int  # chunk 内容长度


@dataclass
class RequestStats:
    """请求统计：记录每个请求的关键指标"""
    user_id: int
    category: str
    rtt: float
    profile_health: float  # 用户配置的健康度（profile.health）
    used_health_factor: float  # 实际使用的健康度（baseline=1.0, network-aware=profile.health）
    ttft: float  # Time To First Chunk（秒）
    total_chunks: int
    total_time: float  # 请求总时间（秒）


def generate_user_profiles(num_users: int = 8192) -> List[UserProfile]:
    """生成用户配置 - 使用正态分布（有长尾）
    
    RTT 分布：
    - 均值 400ms，标准差 1000ms
    - 截断到 [0, 800000] ms 范围（模拟极端情况：有人 RTT 很小，有人很大）
    - 形成长尾分布：大多数用户网络正常，少数用户网络很差
    
    关键：使用 user_id 作为种子，确保与 Hint Server 一致！
    """
    profiles = []
    
    for user_id in range(1, num_users + 1):
        # 使用 user_id 作为种子，确保可重复性
        # 这样无论在哪里调用，user_id=N 的 RTT 都相同
        np.random.seed(user_id + 42)  # +42 作为偏移
        
        # 正态分布 RTT：mean=400ms, std=1000ms
        rtt = np.random.normal(loc=400, scale=1000)
        rtt = float(np.clip(rtt, 0, 800000))
        
        # 根据 RTT 计算健康度：health = exp(-RTT / 500)
        # 使用 500 而不是 150，以适配新的 RTT 分布 (loc=400, scale=1000)
        health = float(np.exp(-rtt / 500.0))
        
        # 根据 RTT 分类
        if rtt >= 400:
            category = 'very_bad'
        elif rtt >= 200:
            category = 'bad'
        elif rtt >= 80:
            category = 'good'
        else:
            category = 'very_good'
        
        profiles.append(UserProfile(
            user_id=user_id,
            rtt=rtt,
            health=health,
            category=category
        ))
    
    # 打乱发送顺序，但 user_id → RTT 的映射保持不变
    np.random.seed(999)
    np.random.shuffle(profiles)
    return profiles


class TimelineExperiment:
    """时间线实验"""
    
    def __init__(
        self,
        vllm_url: str = "http://localhost:8000/v1",
        num_users: int = 1024,
        max_tokens: int = 50,
        concurrency: int = 256,
        client_concurrency: int = 2048,
    ):
        self.vllm_url = vllm_url
        self.num_users = num_users
        self.max_tokens = max_tokens
        self.concurrency = concurrency  # vLLM 的 max_num_seqs
        self.client_concurrency = client_concurrency  # 客户端并发连接数
        self.model_name = None
        
        # 生成用户配置（使用固定种子保证两次实验用户相同）
        np.random.seed(42)
        self.user_profiles = generate_user_profiles(num_users)
        
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
        experiment_start_time: float,
        semaphore: asyncio.Semaphore,
        mode: str,
    ) -> Tuple[List[ChunkEvent], RequestStats]:
        """发送单个请求，记录每个 chunk 的事件和 TTFC
        
        使用 Semaphore 限制客户端并发，避免连接风暴
        但并发数足够大，让 backlog 进入 vLLM 的 waiting 队列
        """
        events = []
        request_start_time = time.perf_counter()
        first_chunk_time = None
        
        prompt = f"User {profile.user_id}: Write a brief story about AI."
        
        # 使用包含 user_id 的格式，让 vLLM 的 _extract_user_id 能识别
        # 格式: user{N}_{random} -> vLLM 可以提取 user_id = N
        import uuid
        custom_request_id = f"user{profile.user_id}_{uuid.uuid4().hex[:8]}"
        
        # 关键：Baseline 模式传 health_factor=1.0，Network-Aware 模式传 profile.health
        health_factor = 1.0 if mode == "baseline" else profile.health
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "stream": True,
            "temperature": 0.0,  # 固定采样，确保可复现
            "top_p": 1.0,  # 固定采样
            "ignore_eos": True,  # 强制生成 max_tokens，避免提前结束
            "user": f"user{profile.user_id}",
            "request_id": custom_request_id,  # vLLM 支持自定义 request_id！
            # 直接传递健康度，避免查询 hint server
            "vllm_xargs": {
                "health_factor": health_factor
            }
        }
        
        # 使用 Semaphore 控制并发
        async with semaphore:
            try:
                async with session.post(
                    f"{self.vllm_url}/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    if resp.status != 200:
                        return events, None
                    
                    chunk_idx = 0
                    
                    async for line in resp.content:
                        if not line:
                            continue
                        
                        line_str = line.decode('utf-8').strip()
                        if not line_str.startswith("data: "):
                            continue
                        
                        data_str = line_str[6:]
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    current_time = time.perf_counter()
                                    
                                    # 记录 TTFC（第一个 chunk 的时间）
                                    if first_chunk_time is None:
                                        first_chunk_time = current_time - request_start_time
                                    
                                    # 观测到的到达时间（客户端收到 SSE chunk 的时间）
                                    # 在 localhost 场景下，这近似等于 GPU 生成时间
                                    observed_arrival_time = current_time - experiment_start_time
                                    
                                    # 合成到达时间：加入网络 RTT 延迟后的"有效到达时间"
                                    # 用于计算有效吞吐（ECPS: Effective Chunks Per Second）
                                    synthetic_arrival_time = observed_arrival_time + (profile.rtt / 1000.0 / 2.0)
                                    
                                    events.append(ChunkEvent(
                                        user_id=profile.user_id,
                                        chunk_idx=chunk_idx,
                                        observed_arrival_time=observed_arrival_time,
                                        synthetic_arrival_time=synthetic_arrival_time,
                                        rtt=profile.rtt,
                                        category=profile.category,
                                        chunk_length=len(content)
                                    ))
                                    chunk_idx += 1
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                pass
        
        # 创建请求统计
        total_time = time.perf_counter() - request_start_time
        # 实际使用的健康度：baseline=1.0, network-aware=profile.health
        used_health_factor = 1.0 if mode == "baseline" else profile.health
        
        stats = RequestStats(
            user_id=profile.user_id,
            category=profile.category,
            rtt=profile.rtt,
            profile_health=profile.health,  # 用户配置的健康度
            used_health_factor=used_health_factor,  # 实际使用的健康度
            ttft=first_chunk_time if first_chunk_time else 0,
            total_chunks=len(events),
            total_time=total_time
        ) if events else None
        
        return events, stats
    
    async def run_experiment(
        self, 
        mode: str, 
        shuffled_profiles: List[UserProfile]
    ) -> Tuple[List[ChunkEvent], List[RequestStats], float]:
        """运行实验，返回所有 chunk 事件和请求统计
        
        Args:
            mode: "baseline" 或 "network_aware"
            shuffled_profiles: 已打乱的用户配置列表（两个实验复用同一个顺序）
        """
        print(f"\n{'='*60}")
        print(f"🚀 Running {mode.upper()} - {self.num_users} Users")
        print(f"{'='*60}")
        
        # 设置 TCPConnector limit，避免 aiohttp 自己无限开连接
        connector = aiohttp.TCPConnector(limit=self.client_concurrency)
        async with aiohttp.ClientSession(connector=connector) as session:
            if not self.model_name:
                self.model_name = await self.detect_model(session)
                print(f"📦 Model: {self.model_name}")
            
            print(f"🔢 Users: {self.num_users}")
            print(f"🎯 vLLM max_num_seqs (assumed): {self.concurrency}")
            print(f"🔌 Client concurrency: {self.client_concurrency}")
            print(f"📝 Max tokens: {self.max_tokens}")
            print(f"⚠️  Note: max_num_seqs must match vLLM server config")
            
            experiment_start_time = time.perf_counter()
            
            # 使用 Semaphore 限制客户端并发连接数
            # 策略：足够大但有限，让 backlog 进入 vLLM 的 waiting 队列
            # 但不会压垮系统（避免连接风暴和 IO 瓶颈）
            semaphore = asyncio.Semaphore(self.client_concurrency)
            
            # 创建所有任务（但受 Semaphore 控制并发执行）
            tasks = [
                self.send_request(session, profile, experiment_start_time, semaphore, mode)
                for profile in shuffled_profiles
            ]
            
            # 计算预期的 waiting 队列规模
            # 由于 Semaphore 限制，同一时刻最多 client_concurrency 个请求进入 vLLM
            # 其中 concurrency 个在 running，其余在 waiting
            expected_backlog = max(0, self.client_concurrency - self.concurrency)
            
            print(f"\n⏳ Sending {len(tasks)} requests...")
            print(f"   Client concurrency: {self.client_concurrency} (Semaphore)")
            print(f"   vLLM max_num_seqs: {self.concurrency} (scheduler limit)")
            print(f"   Expected backlog: ~{expected_backlog} requests in waiting queue")
            print(f"   Health factor: {'1.0 (all users)' if mode == 'baseline' else 'varies by RTT'}")
            results = await asyncio.gather(*tasks)
            
            duration = time.perf_counter() - experiment_start_time
        
        # 合并所有事件和统计
        all_events = []
        all_stats = []
        completed_requests = 0
        failed_requests = 0
        for events, stats in results:
            if stats and len(events) > 0:
                all_events.extend(events)
                all_stats.append(stats)
                completed_requests += 1
            else:
                failed_requests += 1
        
        # 计算 TTFC 统计
        if all_stats:
            ttfts = [s.ttft for s in all_stats if s.ttft > 0]  # ttft 字段名保留，但实际是 TTFC
            if ttfts:
                avg_ttft = np.mean(ttfts)
                p50_ttft = np.percentile(ttfts, 50)
                p95_ttft = np.percentile(ttfts, 95)
                p99_ttft = np.percentile(ttfts, 99)
                
                # Chunk 长度验证（检查是否一 token 一 chunk）
                chunk_lengths = [e.chunk_length for e in all_events]
                if chunk_lengths:
                    avg_chunk_len = np.mean(chunk_lengths)
                    max_chunk_len = max(chunk_lengths)
                    chunk_len_dist = {1: sum(1 for l in chunk_lengths if l == 1),
                                    2: sum(1 for l in chunk_lengths if l == 2),
                                    3: sum(1 for l in chunk_lengths if l == 3),
                                    '>3': sum(1 for l in chunk_lengths if l > 3)}
                
                print(f"📊 Total chunks: {len(all_events)}")
                print(f"✅ Completed requests: {completed_requests}/{len(results)}")
                print(f"❌ Failed requests: {failed_requests}/{len(results)}")
                print(f"⏱️  Duration: {duration:.2f}s")
                
                # Chunk 长度统计（用于验证 chunk 大小分布）
                if chunk_lengths:
                    print(f"\n📏 Chunk Length Statistics:")
                    print(f"   Avg: {avg_chunk_len:.2f} chars")
                    print(f"   Max: {max_chunk_len} chars")
                    print(f"   Distribution: 1 char={chunk_len_dist[1]}, 2 chars={chunk_len_dist[2]}, 3 chars={chunk_len_dist[3]}, >3 chars={chunk_len_dist['>3']}")
                
                print(f"\n⚡ TTFC (Time To First Chunk) Statistics:")
                print(f"   Avg: {avg_ttft*1000:.1f}ms")
                print(f"   P50: {p50_ttft*1000:.1f}ms")
                print(f"   P95: {p95_ttft*1000:.1f}ms")
                print(f"   P99: {p99_ttft*1000:.1f}ms")
                
                # 按类别统计 TTFC
                print(f"\n⚡ TTFC by Category:")
                for cat in ['very_good', 'good', 'bad', 'very_bad']:
                    cat_ttfts = [s.ttft for s in all_stats if s.category == cat and s.ttft > 0]
                    if cat_ttfts:
                        print(f"   {cat:10s}: Avg={np.mean(cat_ttfts)*1000:.1f}ms, P50={np.percentile(cat_ttfts, 50)*1000:.1f}ms, P95={np.percentile(cat_ttfts, 95)*1000:.1f}ms")
        else:
            print(f"📊 Total chunks: {len(all_events)}")
            print(f"⏱️  Duration: {duration:.2f}s")
        
        return all_events, all_stats, duration
    
    def compute_cumulative_curve(
        self, 
        events: List[ChunkEvent], 
        time_points: np.ndarray,
        use_synthetic_arrival: bool = True
    ) -> np.ndarray:
        """计算累计有效 chunk 曲线
        
        Args:
            events: chunk 事件列表
            time_points: 时间采样点
            use_synthetic_arrival: True 使用合成到达时间（加入 RTT），False 使用观测到达时间（≈ GPU 生成时间）
        """
        cumulative = np.zeros(len(time_points))
        
        if use_synthetic_arrival:
            times = sorted([e.synthetic_arrival_time for e in events])
        else:
            times = sorted([e.observed_arrival_time for e in events])
        
        event_idx = 0
        for i, t in enumerate(time_points):
            while event_idx < len(times) and times[event_idx] <= t:
                event_idx += 1
            cumulative[i] = event_idx
        
        return cumulative
    
    async def run_comparison(self):
        """运行对比实验"""
        print("\n" + "🕐" * 30)
        print("     TIMELINE EXPERIMENT")
        print("     验证 Network-Aware 的真正优势")
        print("🕐" * 30)
        
        # 生成用户配置（两次实验用相同的用户）
        np.random.seed(42)
        self.user_profiles = generate_user_profiles(self.num_users)
        
        # 打印用户分布
        categories = {}
        for p in self.user_profiles:
            categories[p.category] = categories.get(p.category, 0) + 1
        print(f"\n📊 用户分布: {categories}")
        print(f"   RTT 范围: {min(p.rtt for p in self.user_profiles):.1f} - {max(p.rtt for p in self.user_profiles):.1f} ms")
        print(f"   Health 范围: {min(p.health for p in self.user_profiles):.3f} - {max(p.health for p in self.user_profiles):.3f}")
        
        # 关键：生成固定的请求到达顺序，两个实验复用
        # 这样 baseline 和 network-aware 的到达顺序完全一致，对比才有意义
        import random
        shuffled_profiles = self.user_profiles.copy()
        random.Random(12345).shuffle(shuffled_profiles)  # 固定种子，保证可复现
        print(f"\n🔀 Request arrival order: Fixed seed (12345) for both experiments")
        
        # Baseline：所有用户 health=1.0，FCFS 调度
        baseline_events, baseline_stats, baseline_duration = await self.run_experiment("baseline", shuffled_profiles)
        
        await asyncio.sleep(3)
        
        # Network-Aware：根据 RTT 计算 health，优先调度健康度高的用户
        network_events, network_stats, network_duration = await self.run_experiment("network_aware", shuffled_profiles)
        
        # 计算曲线
        max_time = max(baseline_duration, network_duration)
        time_points = np.linspace(0, max_time, 500)
        
        # 累计观测到的 chunk（GPU 视角，应该相同）
        # 使用 observed_arrival_time（在 localhost 场景下 ≈ GPU 生成时间）
        baseline_observed = self.compute_cumulative_curve(baseline_events, time_points, use_synthetic_arrival=False)
        network_observed = self.compute_cumulative_curve(network_events, time_points, use_synthetic_arrival=False)
        
        # 累计有效到达的 chunk（客户端视角，加入 RTT 延迟，应该不同！）
        # 使用 synthetic_arrival_time（加入网络 RTT 后的有效到达时间）
        baseline_arrived = self.compute_cumulative_curve(baseline_events, time_points, use_synthetic_arrival=True)
        network_arrived = self.compute_cumulative_curve(network_events, time_points, use_synthetic_arrival=True)
        
        # 绘图
        plt.figure(figsize=(14, 10))
        
        # 子图1：累计观测到的 chunk（GPU 视角，localhost ≈ 生成时间）
        plt.subplot(2, 2, 1)
        plt.plot(time_points, baseline_observed, 'r-', label='Baseline', linewidth=2)
        plt.plot(time_points, network_observed, 'g-', label='Network-Aware', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Cumulative Chunks Observed')
        plt.title('GPU Perspective: Cumulative Chunks Observed\n(localhost ≈ generation time, both should be identical)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2：累计到达的 chunk（客户端视角）
        plt.subplot(2, 2, 2)
        plt.plot(time_points, baseline_arrived, 'r-', label='Baseline', linewidth=2)
        plt.plot(time_points, network_arrived, 'g-', label='Network-Aware', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Cumulative Chunks Arrived')
        plt.title('Client Perspective: Cumulative Chunks Arrived\n(Network-Aware should always be on top!)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图3：差值（Network-Aware - Baseline）
        plt.subplot(2, 2, 3)
        diff = network_arrived - baseline_arrived
        plt.fill_between(time_points, 0, diff, where=(diff > 0), color='green', alpha=0.5, label='Network-Aware 领先')
        plt.fill_between(time_points, 0, diff, where=(diff < 0), color='red', alpha=0.5, label='Baseline 领先')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Chunk Difference')
        plt.title('Difference: Network-Aware - Baseline\n(Positive = Network-Aware leads)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图4：ECPS (Effective Chunks Per Second) 随时间变化
        plt.subplot(2, 2, 4)
        # 避免除以0
        etps_baseline = np.zeros_like(time_points)
        etps_network = np.zeros_like(time_points)
        for i, t in enumerate(time_points):
            if t > 0.5:  # 从0.5秒开始计算
                etps_baseline[i] = baseline_arrived[i] / t
                etps_network[i] = network_arrived[i] / t
        
        plt.plot(time_points, etps_baseline, 'r-', label='Baseline ECPS', linewidth=2)
        plt.plot(time_points, etps_network, 'g-', label='Network-Aware ECPS', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('ECPS (Effective Chunks Per Second)')
        plt.title('ECPS Over Time\n(Network-Aware should always be higher!)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/v-boxiuli/eBPF-TokenFlow/demo/timeline_comparison.png', dpi=150)
        print(f"\n📈 图表已保存到: /home/v-boxiuli/eBPF-TokenFlow/demo/timeline_comparison.png")
        
        # 打印统计
        print("\n" + "=" * 60)
        print("📊 STATISTICAL SUMMARY")
        print("=" * 60)
        
        # 找到中间时间点的数据
        mid_idx = len(time_points) // 2
        mid_time = time_points[mid_idx]
        
        print(f"\n在 t={mid_time:.1f}s 时:")
        print(f"  Baseline 到达: {int(baseline_arrived[mid_idx])} chunks")
        print(f"  Network-Aware 到达: {int(network_arrived[mid_idx])} chunks")
        print(f"  差值: {int(diff[mid_idx])} chunks ({diff[mid_idx]/max(baseline_arrived[mid_idx],1)*100:+.1f}%)")
        
        # 最终结果
        print(f"\n最终结果 (t={max_time:.1f}s):")
        print(f"  Baseline 到达: {int(baseline_arrived[-1])} chunks")
        print(f"  Network-Aware 到达: {int(network_arrived[-1])} chunks")
        print(f"  差值: {int(diff[-1])} chunks")
        
        # 平均领先量
        avg_lead = np.mean(diff)
        print(f"\n平均领先量: {avg_lead:.1f} chunks")
        print(f"领先时间比例: {np.mean(diff > 0)*100:.1f}%")
        
        # 按类别统计
        print(f"\n📊 按用户类别统计:")
        for cat in ['very_bad', 'bad', 'good', 'very_good']:
            b_count = len([e for e in baseline_events if e.category == cat])
            n_count = len([e for e in network_events if e.category == cat])
            # 平均延迟 = synthetic_arrival_time - observed_arrival_time = RTT/2
            b_avg_delay = np.mean([e.synthetic_arrival_time - e.observed_arrival_time for e in baseline_events if e.category == cat]) * 1000 if b_count > 0 else 0
            n_avg_delay = np.mean([e.synthetic_arrival_time - e.observed_arrival_time for e in network_events if e.category == cat]) * 1000 if n_count > 0 else 0
            print(f"  {cat:10s}: Baseline {b_count:5d} chunks (avg delay {b_avg_delay:.0f}ms), Network-Aware {n_count:5d} chunks (avg delay {n_avg_delay:.0f}ms)")
        
        # TTFT 对比统计
        print("\n" + "=" * 60)
        print("⚡ TTFT COMPARISON")
        print("=" * 60)
        
        if baseline_stats and network_stats:
            b_ttfts = [s.ttft for s in baseline_stats if s.ttft > 0]
            n_ttfts = [s.ttft for s in network_stats if s.ttft > 0]
            
            if b_ttfts and n_ttfts:
                print(f"\n📊 Overall TTFT:")
                print(f"  Baseline:      Avg={np.mean(b_ttfts)*1000:.1f}ms, P50={np.percentile(b_ttfts, 50)*1000:.1f}ms, P95={np.percentile(b_ttfts, 95)*1000:.1f}ms, P99={np.percentile(b_ttfts, 99)*1000:.1f}ms")
                print(f"  Network-Aware: Avg={np.mean(n_ttfts)*1000:.1f}ms, P50={np.percentile(n_ttfts, 50)*1000:.1f}ms, P95={np.percentile(n_ttfts, 95)*1000:.1f}ms, P99={np.percentile(n_ttfts, 99)*1000:.1f}ms")
                
                ttft_improvement = (np.mean(b_ttfts) - np.mean(n_ttfts)) / np.mean(b_ttfts) * 100
                print(f"\n  TTFT Improvement: {ttft_improvement:+.1f}%")
                
                # 按类别统计 TTFT
                print(f"\n📊 TTFT by Category:")
                for cat in ['very_good', 'good', 'bad', 'very_bad']:
                    b_cat_ttfts = [s.ttft for s in baseline_stats if s.category == cat and s.ttft > 0]
                    n_cat_ttfts = [s.ttft for s in network_stats if s.category == cat and s.ttft > 0]
                    
                    if b_cat_ttfts and n_cat_ttfts:
                        b_avg = np.mean(b_cat_ttfts) * 1000
                        n_avg = np.mean(n_cat_ttfts) * 1000
                        improvement = (b_avg - n_avg) / b_avg * 100 if b_avg > 0 else 0
                        print(f"  {cat:10s}: Baseline Avg={b_avg:.1f}ms, Network-Aware Avg={n_avg:.1f}ms ({improvement:+.1f}%)")


async def main():
    parser = argparse.ArgumentParser(description="Timeline Experiment")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--num-users", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=256, 
                       help="vLLM max_num_seqs (scheduler limit)")
    parser.add_argument("--client-concurrency", type=int, default=2048,
                       help="Client-side concurrency limit (Semaphore)")
    args = parser.parse_args()
    
    experiment = TimelineExperiment(
        vllm_url=args.vllm_url,
        num_users=args.num_users,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        client_concurrency=args.client_concurrency,
    )
    
    await experiment.run_comparison()


if __name__ == "__main__":
    asyncio.run(main())

