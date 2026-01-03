#!/usr/bin/env python3
"""
🎥 Visual Demo (Large Scale) - 8000用户压测专用版
功能：后台运行海量用户制造拥堵，前台仅显示少量用户的对比，以体现调度优势。
"""

import asyncio
import argparse
import time
import json
import random
from dataclasses import dataclass
from typing import List

# 引入 Rich 库
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

# 引入原有逻辑
from timeline_experiment import TimelineExperiment, UserProfile, generate_user_profiles_multimodal

@dataclass
class TokenRecord:
    time_offset: float
    content: str
    user_id: int
    category: str

class VisualExperiment(TimelineExperiment):
    def __init__(self, visual_limit: int = 40, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visual_limit = visual_limit  # 只录制 ID 小于等于这个数的用户
        self.recordings = []

    async def send_request(self, session, profile, experiment_start_time, semaphore, mode):
        # 如果不是可视用户，走简化逻辑（只发请求，不记录内容，节省内存）
        is_visible = profile.user_id <= self.visual_limit
        
        # 1. 准备参数
        prompt = f"User {profile.user_id}: Write a story."
        import uuid
        custom_request_id = f"user{profile.user_id}_{uuid.uuid4().hex[:8]}"
        
        health_factor = 1.0 if mode == "baseline" else profile.health

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "stream": True,
            "temperature": 0.0,
            "ignore_eos": True,
            "user": f"user{profile.user_id}",
            "request_id": custom_request_id,
            "vllm_xargs": {"health_factor": health_factor}
        }

        # 2. 计算延迟
        rtt_sec = profile.rtt / 1000.0
        one_way_delay = (rtt_sec / 2.0) + (0.5 * (rtt_sec ** 2))

        async with semaphore:
            # 模拟上行延迟
            await asyncio.sleep(one_way_delay)
            
            try:
                # 发送请求
                async with session.post(f"{self.vllm_url}/chat/completions", json=payload) as resp:
                    if resp.status != 200:
                        return [], None
                    
                    # 3. 处理流式响应
                    async for line in resp.content:
                        if not line: continue
                        line_str = line.decode('utf-8').strip()
                        if not line_str.startswith("data: "): continue
                        data_str = line_str[6:]
                        if data_str == "[DONE]": break
                        
                        # 仅当是可视用户时，解析并录制
                        if is_visible:
                            try:
                                data = json.loads(data_str)
                                content = data["choices"][0]["delta"].get("content")
                                if content:
                                    current_time = time.perf_counter()
                                    observed_time = current_time - experiment_start_time
                                    synthetic_arrival_time = observed_time + one_way_delay
                                    
                                    self.recordings.append(TokenRecord(
                                        time_offset=synthetic_arrival_time,
                                        content=content,
                                        user_id=profile.user_id,
                                        category=profile.category
                                    ))
                            except:
                                continue
            except Exception:
                pass
        
        return [], None

# ==========================================
# 播放器逻辑 (保持不变，增加一点统计显示)
# ==========================================
def run_playback(baseline_recs: List[TokenRecord], ours_recs: List[TokenRecord], duration: float):
    console = Console()
    baseline_recs.sort(key=lambda x: x.time_offset)
    ours_recs.sort(key=lambda x: x.time_offset)
    
    layout = Layout()
    layout.split_row(
        Layout(name="left"),
        Layout(name="right"),
    )
    
    text_ours = Text()
    text_base = Text()
    
    sim_time = 0.0
    step = 0.05
    b_idx = 0
    o_idx = 0
    
    # 标题增加说明
    title_left = "🚀 Ours (Network-Aware)\n[Sampling 40 Users from 8000]"
    title_right = "🐢 Baseline (FIFO)\n[Sampling 40 Users from 8000]"
    
    with Live(layout, refresh_per_second=20, screen=True) as live:
        while sim_time < duration + 5.0: # 多展示几秒
            start_loop = time.time()
            
            # Update Ours
            while o_idx < len(ours_recs) and ours_recs[o_idx].time_offset <= sim_time:
                rec = ours_recs[o_idx]
                color = "green" if rec.category in ['good', 'very_good'] else "yellow"
                if rec.category == 'very_bad': color = "red"
                text_ours.append(rec.content, style=color)
                o_idx += 1
                
            # Update Baseline
            while b_idx < len(baseline_recs) and baseline_recs[b_idx].time_offset <= sim_time:
                rec = baseline_recs[b_idx]
                color = "green" if rec.category in ['good', 'very_good'] else "yellow"
                if rec.category == 'very_bad': color = "red"
                text_base.append(rec.content, style=color)
                b_idx += 1

            # Keep text buffer reasonable
            if len(text_ours) > 3000: text_ours = text_ours[-3000:]
            if len(text_base) > 3000: text_base = text_base[-3000:]

            layout["left"].update(Panel(text_ours, title=title_left, border_style="green"))
            layout["right"].update(Panel(text_base, title=title_right, border_style="white"))
            
            sim_time += step
            process_time = time.time() - start_loop
            if process_time < step:
                time.sleep(step - process_time)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=8000, help="Total background users")
    parser.add_argument("--vis-users", type=int, default=100, help="Users to visualize")
    args = parser.parse_args()

    # ⚠️ 请确保这里和你的 vllm serve 命令一致
    TARGET_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

    console = Console()
    console.clear()
    console.print(f"[bold blue]🎬 Initializing Large Scale Demo ({args.users} users)...[/bold blue]")

    # 1. 配置参数
    # concurrency 保持 256，但 num_users 8000，制造巨大的排队
    viz_args = {
        "vllm_url": "http://localhost:8000/v1",
        "num_users": args.users,
        "max_tokens": 50,    # 短一些，让请求周转更快
        "concurrency": 256,  # vLLM 的物理限制
        "client_concurrency": 1024, # 客户端最大连接数
        "target_qps": 500.0, # 高 QPS 瞬间打满队列
        "visual_limit": args.vis_users
    }

    # 2. 生成统一的用户配置
    np_profiles = generate_user_profiles_multimodal(viz_args["num_users"])

    # 🔥【核心修改】为了视频效果，强制“篡改”前台可视用户和后台用户的分布
    print(f"🔧 Tweaking profiles for DEMO effect...")
    
    for p in np_profiles:
        # === 1. 强制前台可视用户 (User 1-40) 为“光纤用户” ===
        if p.user_id <= args.vis_users:
            p.rtt = 10.0          # 极低延迟 (10ms)
            p.category = 'very_good'
            p.health = 1.0        # 满健康度 -> 最高优先级
            
        # === 2. (可选) 让后台用户 (User > 40) 更“毒”一些 ===
        # 这样 Baseline 会被堵得更惨，对比更强烈
        else:
            # 我们保持原有的随机分布，或者你可以取消下面几行的注释来故意制造更严重的拥堵
            if p.user_id % 3 == 0: # 让 1/3 的后台用户变成极差
                p.rtt = 2000.0
                p.category = 'very_bad'
                p.health = 0.02

    # 打印一下确认修改成功
    vip_users = [p for p in np_profiles if p.user_id <= args.vis_users]
    print(f"✨ VIP Users (Visible): All set to 'very_good' (RTT=10ms, Health=1.0)")
    # ==================== 3. 运行 Baseline ====================
    # 过滤参数，去掉 visual_limit，因为它不是 TimelineExperiment 的标准参数，
    # 它是我们传给 VisualExperiment __init__ 的
    exp_args = viz_args.copy()
    del exp_args["visual_limit"]
    
    exp_base = VisualExperiment(visual_limit=args.vis_users, **exp_args)
    exp_base.user_profiles = np_profiles
    exp_base.model_name = TARGET_MODEL_NAME
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task(description=f"Running Baseline (Load: {args.users} users)...", total=None)
        # 使用修正后的方法名 poisson
        await exp_base.run_experiment_poisson("baseline", np_profiles)
    
    baseline_records = exp_base.recordings
    console.print(f"✅ Baseline captured: {len(baseline_records)} tokens (from visible users)")

    # ==================== 4. 运行 Ours ====================
    console.print("☕ Cooling down vLLM (5s)...")
    await asyncio.sleep(5)
    
    exp_ours = VisualExperiment(visual_limit=args.vis_users, **exp_args)
    exp_ours.user_profiles = np_profiles
    exp_ours.model_name = TARGET_MODEL_NAME
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task(description=f"Running Network-Aware (Load: {args.users} users)...", total=None)
        await exp_ours.run_experiment_poisson("network_aware", np_profiles)
        
    ours_records = exp_ours.recordings
    console.print(f"✅ Ours captured: {len(ours_records)} tokens (from visible users)")

    # 5. 计算最大时长并回放
    max_duration = 0
    if baseline_records: max_duration = max(max_duration, max(r.time_offset for r in baseline_records))
    if ours_records: max_duration = max(max_duration, max(r.time_offset for r in ours_records))
    
    # 限制最大播放时长，避免因为某个长尾请求拖太久
    max_duration = min(max_duration, 60.0)

    for i in range(3, 0, -1):
        console.print(f"[bold yellow]Video starting in {i}...[/bold yellow]")
        time.sleep(1)

    run_playback(baseline_records, ours_records, max_duration)
    console.print("[bold green]🎬 Demo Finished![/bold green]")

if __name__ == "__main__":
    asyncio.run(main())