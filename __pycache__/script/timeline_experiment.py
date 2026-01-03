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

def generate_user_profiles_multimodal(num_users: int = 8192) -> List[UserProfile]:
    """改进版：使用混合高斯分布模拟真实的 4 类用户群体"""
    profiles = []
    
    # 定义 4 种网络环境的参数 (概率, 均值ms, 标准差ms, 类别名)
    # 概率总和应为 1.0
    network_clusters = [
        # 1. 极好网络 (光纤/同城): 提升到 50%
        {'prob': 0.50, 'loc': 20,  'scale': 10,  'cat': 'very_good'},
        
        # 2. 普通网络 (4G/Wi-Fi): 提升到 40%
        {'prob': 0.40, 'loc': 200, 'scale': 30,  'cat': 'good'},
        
        # 3. 较差网络 (跨国): 降到 9% (作为边缘案例)
        {'prob': 0.09, 'loc': 700, 'scale': 80,  'cat': 'bad'},
        
        # 4. 极差网络 (卫星): 降到 1% (作为极端案例)
        {'prob': 0.01, 'loc': 2000,'scale': 400, 'cat': 'very_bad'}
    ]
    # network_clusters = [
    #     # 1. 极好网络 (光纤/同城): 约占 40%
    #     {'prob': 0.20, 'loc': 20,  'scale': 10,  'cat': 'very_good'},
        
    #     # 2. 普通网络 (4G/Wi-Fi): 约占 30%
    #     {'prob': 0.70, 'loc': 200, 'scale': 30,  'cat': 'good'},
        
    #     # 3. 较差网络 (跨国/拥堵): 约占 20%
    #     {'prob': 0.05, 'loc': 700, 'scale': 80,  'cat': 'bad'},
        
    #     # 4. 极差网络 (弱信号/卫星): 约占 10%
    #     {'prob': 0.05, 'loc': 2000,'scale': 400, 'cat': 'very_bad'}
    # ]
    
    for user_id in range(1, num_users + 1):
        np.random.seed(user_id + 42)
        
        # 1. 先决定这个用户属于哪个群体
        cluster_idx = np.random.choice(
            len(network_clusters), 
            p=[c['prob'] for c in network_clusters]
        )
        cluster = network_clusters[cluster_idx]
        
        # 2. 在该群体的分布内生成 RTT
        rtt = np.random.normal(loc=cluster['loc'], scale=cluster['scale'])
        
        # 3. 物理限制修正（RTT 不能小于 5ms，不能无限大）
        rtt = float(np.clip(rtt, 5, 10000))
        
        # 4. 计算健康度 (沿用原逻辑)
        # 注意：这里 health 计算可能需要根据不同群体的 RTT 范围做微调，
        # 或者继续使用统一的衰减公式
        health = float(np.exp(-rtt / 500.0))
        
        profiles.append(UserProfile(
            user_id=user_id,
            rtt=rtt,
            health=health,
            category=cluster['cat'] # 直接使用生成的类别
        ))
    
    # 打乱顺序，模拟真实到达
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
        target_qps: float = 20.0,
    ):
        self.vllm_url = vllm_url
        self.num_users = num_users
        self.max_tokens = max_tokens
        self.concurrency = concurrency  # vLLM 的 max_num_seqs
        self.client_concurrency = client_concurrency  # 客户端并发连接数
        self.target_qps = target_qps
        self.model_name = None
        
        # 生成用户配置（使用固定种子保证两次实验用户相同）
        np.random.seed(42)
        self.user_profiles = generate_user_profiles_multimodal(num_users)
        
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
        
        修改点：
        1. 模拟上行延迟：请求发送前 sleep (RTT/2 + 惩罚)
        2. 模拟下行延迟：接收数据后 add (RTT/2 + 惩罚)
        """
        events = []
        # 记录请求开始处理的时间（Client 决定发送的时间）
        request_start_time = time.perf_counter()
        first_chunk_time = None
        
        prompt = f"User {profile.user_id}: Write a brief story about AI."
        
        import uuid
        custom_request_id = f"user{profile.user_id}_{uuid.uuid4().hex[:8]}"
        
        # ------------------------------------------------------------------
        # 1. 优先级/健康度计算
        # ------------------------------------------------------------------
        if mode == "baseline":
            health_factor = 1.0
        else:
            # health_factor = 1.0 - profile.health 
            health_factor = profile.health # 原版
            
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "stream": True,
            "temperature": 0.0,
            "top_p": 1.0,
            "ignore_eos": True,
            "user": f"user{profile.user_id}",
            "request_id": custom_request_id,
            "vllm_xargs": {
                "health_factor": health_factor
            }
        }
        
        # ------------------------------------------------------------------
        # 2. 计算单向延迟 (One-Way Delay)
        # ------------------------------------------------------------------
        # 逻辑：物理传输时间(RTT/2) + 拥塞惩罚(RTT^2 的一半)
        # 这样 上行+下行 的总延迟 ≈ RTT + RTT^2
        rtt_sec = profile.rtt / 1000.0
        one_way_delay = (rtt_sec / 2.0) + (0.5 * (rtt_sec ** 2))
        
        async with semaphore:
            # --------------------------------------------------------------
            # 3. 模拟上行延迟 (Uplink Latency)
            # --------------------------------------------------------------
            # 请求在路上跑，还没到 vLLM
            await asyncio.sleep(one_way_delay)
            
            try:
                # 这里的 session.post 发生时刻，相当于 Server 收到请求的时刻
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
                                    
                                    # TTFC 计算：包含了 上行延迟 + 排队 + 生成 + 网络传输(如果是真实网络)
                                    if first_chunk_time is None:
                                        first_chunk_time = current_time - request_start_time
                                    
                                    # -------------------------------------------------------
                                    # 4. 观测时间 & 下行延迟
                                    # -------------------------------------------------------
                                    # observed: 实际上因为前面 sleep 了上行时间，
                                    # 这个 observed 时间已经包含了 (上行 + GPU处理)
                                    observed_arrival_time = current_time - experiment_start_time
                                    
                                    # synthetic: 在 observed 基础上再加一段回去的路程 (下行)
                                    synthetic_arrival_time = observed_arrival_time + one_way_delay
                                    
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
        used_health_factor = 1.0 if mode == "baseline" else profile.health
        
        stats = RequestStats(
            user_id=profile.user_id,
            category=profile.category,
            rtt=profile.rtt,
            profile_health=profile.health,
            used_health_factor=used_health_factor,
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
    
    async def run_experiment_poisson(
        self, 
        mode: str, 
        shuffled_profiles: List[UserProfile]
    ) -> Tuple[List[ChunkEvent], List[RequestStats], float]:
        """运行实验 (Poisson Arrival Mode)"""
        print(f"\n{'='*60}")
        print(f"🚀 Running {mode.upper()} - {self.num_users} Users")
        print(f"🌊 Mode: Poisson Arrival Process (Target QPS: {self.target_qps})")
        print(f"{'='*60}")
        
        # 1. 预先计算泊松到达时间
        # 泊松过程的间隔时间服从指数分布
        # scale = 1 / lambda (QPS)
        np.random.seed(12345) # 固定种子，保证两种模式下的到达时间完全一致
        inter_arrival_times = np.random.exponential(1.0 / self.target_qps, len(shuffled_profiles))
        
        # 计算每个请求相对于实验开始的绝对发射时间
        scheduled_start_times = np.cumsum(inter_arrival_times)
        total_expected_duration = scheduled_start_times[-1]
        
        connector = aiohttp.TCPConnector(limit=self.client_concurrency)
        async with aiohttp.ClientSession(connector=connector) as session:
            if not self.model_name:
                self.model_name = await self.detect_model(session)
                print(f"📦 Model: {self.model_name}")
            
            print(f"   vLLM max_num_seqs: {self.concurrency}")
            print(f"   Est. Request Injection Duration: {total_expected_duration:.2f}s")
            
            experiment_start_time = time.perf_counter()
            
            # 依然保留 Semaphore 作为安全网，防止系统文件句柄耗尽
            # 但主要流量控制由 sleep 决定
            semaphore = asyncio.Semaphore(self.client_concurrency)
            
            tasks = []
            
            # 2. 循环发送请求
            for i, profile in enumerate(shuffled_profiles):
                # 计算需要等待的时间
                now = time.perf_counter() - experiment_start_time
                wait_time = scheduled_start_times[i] - now
                
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                
                # 发射请求 (Fire and Forget)
                # 使用 create_task 将其放入后台运行，主循环继续处理下一个
                task = asyncio.create_task(
                    self.send_request(session, profile, experiment_start_time, semaphore, mode)
                )
                tasks.append(task)
                
                # 简单的进度打印
                if (i + 1) % 100 == 0:
                    sys.stdout.write(f"\r📤 Sent {i + 1}/{len(shuffled_profiles)} requests...")
                    sys.stdout.flush()

            print(f"\n✅ All {len(tasks)} requests sent. Waiting for completion...")
            
            # 3. 等待所有后台任务完成
            results = await asyncio.gather(*tasks)
            
            duration = time.perf_counter() - experiment_start_time

        # --- 以下统计逻辑保持不变 ---
        
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
            ttfts = [s.ttft for s in all_stats if s.ttft > 0]
            if ttfts:
                avg_ttft = np.mean(ttfts)
                p50_ttft = np.percentile(ttfts, 50)
                p95_ttft = np.percentile(ttfts, 95)
                p99_ttft = np.percentile(ttfts, 99)
                
                chunk_lengths = [e.chunk_length for e in all_events]
                if chunk_lengths:
                    avg_chunk_len = np.mean(chunk_lengths)
                    max_chunk_len = max(chunk_lengths)
                
                print(f"📊 Total chunks: {len(all_events)}")
                print(f"✅ Completed requests: {completed_requests}/{len(results)}")
                print(f"❌ Failed requests: {failed_requests}/{len(results)}")
                print(f"⏱️  Actual Duration: {duration:.2f}s")
                
                print(f"\n⚡ TTFC (Time To First Chunk) Statistics:")
                print(f"   Avg: {avg_ttft*1000:.1f}ms")
                print(f"   P50: {p50_ttft*1000:.1f}ms")
                print(f"   P95: {p95_ttft*1000:.1f}ms")
                print(f"   P99: {p99_ttft*1000:.1f}ms")
                
                print(f"\n⚡ TTFC by Category:")
                for cat in ['very_good', 'good', 'bad', 'very_bad']:
                    cat_ttfts = [s.ttft for s in all_stats if s.category == cat and s.ttft > 0]
                    if cat_ttfts:
                        print(f"   {cat:10s}: Avg={np.mean(cat_ttfts)*1000:.1f}ms, P50={np.percentile(cat_ttfts, 50)*1000:.1f}ms")
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
        """运行对比实验：生成两份报告（全部用户 vs 核心用户）和两张图表"""
        print("\n" + "🕐" * 30)
        print("     TIMELINE EXPERIMENT")
        print("     验证 Network-Aware 的真正优势 (Dual Report Mode)")
        print("🕐" * 30)
        
        # 1. 准备实验环境
        np.random.seed(42)
        # 使用你确认过的双峰分布生成函数
        self.user_profiles = generate_user_profiles_multimodal(self.num_users)
        
        # 打印用户分布
        categories = {}
        for p in self.user_profiles:
            categories[p.category] = categories.get(p.category, 0) + 1
        print(f"\n📊 用户分布: {categories}")
        
        # 固定请求顺序
        import random
        shuffled_profiles = self.user_profiles.copy()
        random.Random(12345).shuffle(shuffled_profiles)
        print(f"🔀 Request arrival order: Fixed seed (12345)")
        
        # 2. 运行两轮实验
        baseline_events, baseline_stats, baseline_duration = await self.run_experiment_poisson("baseline", shuffled_profiles)
        await asyncio.sleep(2)
        network_events, network_stats, network_duration = await self.run_experiment_poisson("network_aware", shuffled_profiles)

        # =================================================================================
        # 内部辅助函数：打印统计报告 (已修复缺失的中值/最终值统计)
        # =================================================================================
        def print_report(title, b_events, n_events, b_stats, n_stats, duration, category_filter=None):
            print("\n" + "=" * 60)
            print(f"📊 REPORT: {title}")
            print("=" * 60)
            
            if not b_events or not n_events:
                print("No events to report.")
                return

            # --- [核心修复] 重新计算曲线以获取时间切片数据 ---
            time_points = np.linspace(0, duration, 500)
            b_arr = self.compute_cumulative_curve(b_events, time_points, use_synthetic_arrival=True)
            n_arr = self.compute_cumulative_curve(n_events, time_points, use_synthetic_arrival=True)
            diff = n_arr - b_arr
            
            # 1. 中间点统计 (t=50%)
            mid_idx = len(time_points) // 2
            mid_time = time_points[mid_idx]
            print(f"\n📍 在 t={mid_time:.1f}s 时 (Mid-point):")
            print(f"   Baseline Arrive: {int(b_arr[mid_idx])} chunks")
            print(f"   Network-Aware Arrive: {int(n_arr[mid_idx])} chunks")
            pct_diff = diff[mid_idx]/max(b_arr[mid_idx], 1)*100
            print(f"   差值: {int(diff[mid_idx])} chunks ({pct_diff:+.1f}%)")

            # 2. 最终点统计 (t=100%)
            print(f"\n🏁 最终结果 (t={duration:.1f}s):")
            print(f"   Baseline Arrive: {int(b_arr[-1])} chunks")
            print(f"   Network-Aware Arrive: {int(n_arr[-1])} chunks")
            print(f"   差值: {int(diff[-1])} chunks")

            # 3. 平均领先量
            avg_lead = np.mean(diff)
            lead_time_pct = np.mean(diff > 0) * 100
            print(f"\n📈 整体趋势:")
            print(f"   平均领先量: {avg_lead:.1f} chunks")
            print(f"   领先时间比例: {lead_time_pct:.1f}%")

            # 4. TTFT 统计
            b_ttfts = [s.ttft for s in b_stats if s.ttft > 0]
            n_ttfts = [s.ttft for s in n_stats if s.ttft > 0]
            
            if b_ttfts and n_ttfts:
                print(f"\n⚡ TTFT (Time To First Token) Statistics:")
                print(f"   Baseline:      Avg={np.mean(b_ttfts)*1000:.1f}ms, P99={np.percentile(b_ttfts, 99)*1000:.1f}ms")
                print(f"   Network-Aware: Avg={np.mean(n_ttfts)*1000:.1f}ms, P99={np.percentile(n_ttfts, 99)*1000:.1f}ms")
                improv = (np.mean(b_ttfts) - np.mean(n_ttfts)) / np.mean(b_ttfts) * 100
                print(f"   >>> TTFT Improvement: {improv:+.1f}%")

            # 5. 按类别细分 (如果是全量报告)
            if category_filter is None:
                print(f"\n📦 按用户类别统计 (Chunks & Avg Delay):")
                categories = ['very_bad', 'bad', 'good', 'very_good']
                for cat in categories:
                    # 统计 chunk 数量
                    b_count = len([e for e in b_events if e.category == cat])
                    n_count = len([e for e in n_events if e.category == cat])
                    
                    # 统计平均延迟代价 (Synthetic - Observed)
                    b_delays = [e.synthetic_arrival_time - e.observed_arrival_time for e in b_events if e.category == cat]
                    n_delays = [e.synthetic_arrival_time - e.observed_arrival_time for e in n_events if e.category == cat]
                    
                    b_avg_d = np.mean(b_delays)*1000 if b_delays else 0
                    n_avg_d = np.mean(n_delays)*1000 if n_delays else 0
                    
                    print(f"   {cat:10s}: Baseline {b_count:6d} chunks ({b_avg_d:4.0f}ms), Network-Aware {n_count:6d} chunks ({n_avg_d:4.0f}ms)")

        # =================================================================================
        # 内部辅助函数：绘制图表 (保持不变)
        # =================================================================================
        def plot_chart(title_prefix, filename, b_events, n_events, plot_max_time):
            time_points = np.linspace(0, plot_max_time, 500)
            
            # GPU 视角
            b_obs = self.compute_cumulative_curve(b_events, time_points, use_synthetic_arrival=False)
            n_obs = self.compute_cumulative_curve(n_events, time_points, use_synthetic_arrival=False)
            
            # 客户端视角
            b_arr = self.compute_cumulative_curve(b_events, time_points, use_synthetic_arrival=True)
            n_arr = self.compute_cumulative_curve(n_events, time_points, use_synthetic_arrival=True)
            
            plt.figure(figsize=(14, 10))
            
            # Subplot 1
            plt.subplot(2, 2, 1)
            plt.plot(time_points, b_obs, 'r-', label='Baseline', linewidth=2)
            plt.plot(time_points, n_obs, 'g-', label='Network-Aware', linewidth=2)
            plt.ylabel('Cumulative Chunks Observed')
            plt.title(f'GPU Perspective: {title_prefix}\n')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Subplot 2
            plt.subplot(2, 2, 2)
            plt.plot(time_points, b_arr, 'r-', label='Baseline', linewidth=2)
            plt.plot(time_points, n_arr, 'g-', label='Network-Aware', linewidth=2)
            plt.ylabel('Cumulative Chunks Arrived')
            plt.title(f'Client Perspective: {title_prefix}\n')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Subplot 3
            plt.subplot(2, 2, 3)
            diff = n_arr - b_arr
            plt.fill_between(time_points, 0, diff, where=(diff > 0), color='green', alpha=0.5, label='Network-Aware Leads')
            plt.fill_between(time_points, 0, diff, where=(diff < 0), color='red', alpha=0.5, label='Baseline Leads')
            plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            plt.ylabel('Chunk Difference')
            plt.title('Performance Gap')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Subplot 4
            plt.subplot(2, 2, 4)
            etps_b = np.zeros_like(time_points)
            etps_n = np.zeros_like(time_points)
            for i, t in enumerate(time_points):
                if t > 0.5:
                    etps_b[i] = b_arr[i] / t
                    etps_n[i] = n_arr[i] / t
            plt.plot(time_points, etps_b, 'r-', label='Baseline ECPS', linewidth=2)
            plt.plot(time_points, etps_n, 'g-', label='Network-Aware ECPS', linewidth=2)
            plt.ylabel('ECPS (Chunks/s)')
            plt.xlabel('Time (s)')
            plt.title('Effective Throughput')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = f'/home/argustest/eBPF-TokenFlow/{filename}'
            plt.savefig(save_path, dpi=150)
            print(f"📈 Chart saved to: {save_path}")
            plt.close()

        # =================================================================================
        # 3. 输出第一份结果：全部用户 (All Users)
        # =================================================================================
        print_report("ALL USERS (Full Dataset)", 
                     baseline_events, network_events, 
                     baseline_stats, network_stats, 
                     max(baseline_duration, network_duration))
        
        plot_chart("ALL USERS", "timeline_comparison_all.png", 
                   baseline_events, network_events, 
                   max(baseline_duration, network_duration))

        # =================================================================================
        # 4. 输出第二份结果：核心用户 (Core Users / Good Users Only)
        # =================================================================================
        core_categories = ['very_good', 'good']
        
        # 过滤数据
        b_events_core = [e for e in baseline_events if e.category in core_categories]
        n_events_core = [e for e in network_events if e.category in core_categories]
        b_stats_core = [s for s in baseline_stats if s.category in core_categories]
        n_stats_core = [s for s in network_stats if s.category in core_categories]
        
        # 确定核心用户的时间轴终点
        if n_events_core:
            core_max_time = max(e.synthetic_arrival_time for e in n_events_core) * 1.1
        else:
            core_max_time = max(baseline_duration, network_duration)
        
        print_report(f"CORE USERS ONLY ({core_categories})", 
                     b_events_core, n_events_core, 
                     b_stats_core, n_stats_core, 
                     core_max_time, category_filter=core_categories)
        
        plot_chart("CORE USERS ONLY (Top ~90%)", "timeline_comparison_core.png", 
                   b_events_core, n_events_core, 
                   core_max_time)

async def main():
    parser = argparse.ArgumentParser(description="Timeline Experiment")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--num-users", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=256, 
                       help="vLLM max_num_seqs (scheduler limit)")
    parser.add_argument("--client-concurrency", type=int, default=2048,
                       help="Client-side concurrency limit (Semaphore)")
    parser.add_argument("--qps", type=float, default=50.0, help="Target requests per second")
    args = parser.parse_args()
    
    experiment = TimelineExperiment(
        vllm_url=args.vllm_url,
        num_users=args.num_users,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        client_concurrency=args.client_concurrency,
        target_qps=args.qps,
    )
    
    await experiment.run_comparison()


if __name__ == "__main__":
    asyncio.run(main())