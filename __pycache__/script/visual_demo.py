#!/usr/bin/env python3
"""
🎥 Visual Demo - 视频录制专用脚本
功能：运行实验并记录 Token 到达时间，然后双屏同步回放，展示 Ours vs Baseline 的流畅度差异。
"""

import asyncio
import json
import argparse
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Any

# 引入 Rich 库做 UI
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

# 引入原有逻辑
from timeline_experiment import TimelineExperiment, UserProfile, generate_user_profiles_multimodal

# ==========================================
# 数据结构：用于回放
# ==========================================
@dataclass
class TokenRecord:
    time_offset: float  # 相对于请求开始的时间
    content: str        # Token 内容
    user_id: int
    category: str

# ==========================================
# 继承并改造原实验类，增加“录像”功能
# ==========================================
class VisualExperiment(TimelineExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recordings = []  # 存储所有的 TokenRecord

    async def send_request(self, session, profile, experiment_start_time, semaphore, mode):
        # 复用原有逻辑，但在接收到 Token 时进行“录像”
        # 为了不破坏原有逻辑的复杂性，我们将大部分代码复制并注入钩子
        # (这里必须重写 send_request 以捕获 content，因为原版只记录了时间)
        
        events = []
        request_start_time = time.perf_counter()
        
        # 简单 prompt
        prompt = f"User {profile.user_id}: Write a story."
        import uuid
        custom_request_id = f"user{profile.user_id}_{uuid.uuid4().hex[:8]}"
        
        # 核心：设置 health_factor
        if mode == "baseline":
            health_factor = 1.0
        else:
            health_factor = profile.health

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

        # 计算延迟
        rtt_sec = profile.rtt / 1000.0
        one_way_delay = (rtt_sec / 2.0) + (0.5 * (rtt_sec ** 2))

        async with semaphore:
            # 模拟上行延迟
            await asyncio.sleep(one_way_delay)
            
            try:
                async with session.post(f"{self.vllm_url}/chat/completions", json=payload) as resp:
                    if resp.status != 200:
                        return [], None
                    
                    async for line in resp.content:
                        if not line: continue
                        line_str = line.decode('utf-8').strip()
                        if not line_str.startswith("data: "): continue
                        data_str = line_str[6:]
                        if data_str == "[DONE]": break
                        
                        try:
                            data = json.loads(data_str)
                            content = data["choices"][0]["delta"].get("content")
                            
                            if content:
                                current_time = time.perf_counter()
                                observed_time = current_time - experiment_start_time
                                # 计算客户端视角的到达时间 (Observed + Downlink Delay)
                                synthetic_arrival_time = observed_time + one_way_delay
                                
                                # 🔥 录制 Token 🔥
                                self.recordings.append(TokenRecord(
                                    time_offset=synthetic_arrival_time,
                                    content=content,
                                    user_id=profile.user_id,
                                    category=profile.category
                                ))
                                
                        except:
                            continue
            except:
                pass
        
        # 返回空以免影响流程，只需 recording
        return [], None

# ==========================================
# 播放器逻辑
# ==========================================
def run_playback(baseline_recs: List[TokenRecord], ours_recs: List[TokenRecord], duration: float):
    console = Console()
    
    # 按照时间排序
    baseline_recs.sort(key=lambda x: x.time_offset)
    ours_recs.sort(key=lambda x: x.time_offset)
    
    # 定义布局
    layout = Layout()
    layout.split_row(
        Layout(name="left"),
        Layout(name="right"),
    )
    
    # 文本缓冲区
    text_ours = Text()
    text_base = Text()
    
    # 模拟时间步长
    sim_time = 0.0
    step = 0.05  # 刷新率 20fps
    
    b_idx = 0
    o_idx = 0
    
    # 创建 Live Context
    with Live(layout, refresh_per_second=20, screen=True) as live:
        while sim_time < duration + 2.0: # 多展示2秒
            start_loop = time.time()
            
            # --- 更新 Ours (左边) ---
            while o_idx < len(ours_recs) and ours_recs[o_idx].time_offset <= sim_time:
                rec = ours_recs[o_idx]
                # 用颜色区分网络状况：红色=差，绿色=好
                color = "green" if rec.category in ['good', 'very_good'] else "yellow"
                if rec.category == 'very_bad': color = "red"
                
                text_ours.append(rec.content, style=color)
                o_idx += 1
                
            # --- 更新 Baseline (右边) ---
            while b_idx < len(baseline_recs) and baseline_recs[b_idx].time_offset <= sim_time:
                rec = baseline_recs[b_idx]
                color = "green" if rec.category in ['good', 'very_good'] else "yellow"
                if rec.category == 'very_bad': color = "red"
                
                text_base.append(rec.content, style=color)
                b_idx += 1

            # --- 裁剪文本防止溢出 (只保留最近的 N 个字符) ---
            max_len = 2000
            if len(text_ours) > max_len: text_ours = text_ours[-max_len:]
            if len(text_base) > max_len: text_base = text_base[-max_len:]

            # --- 更新面板 ---
            layout["left"].update(
                Panel(text_ours, title="🚀 Ours (Network-Aware)", border_style="green", padding=(1, 1))
            )
            layout["right"].update(
                Panel(text_base, title="🐢 Baseline (FIFO)", border_style="white", padding=(1, 1))
            )
            
            # 推进时间
            sim_time += step
            
            # 保持帧率
            process_time = time.time() - start_loop
            if process_time < step:
                time.sleep(step - process_time)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=50, help="Demo user count")
    args = parser.parse_args()
    
    # ⚠️ 必须与你 vllm serve 启动时的模型名称完全一致
    # 你的启动命令是: vllm serve Qwen/Qwen3-4B-Instruct-2507 ...
    TARGET_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

    console = Console()
    console.clear()
    console.print("[bold blue]🎬 Initializing Visual Demo...[/bold blue]")

    # 1. 配置参数
    viz_args = {
        "vllm_url": "http://localhost:8000/v1",  # 确保端口正确
        "num_users": args.users,
        "max_tokens": 100,
        "concurrency": 256,
        "client_concurrency": 256,
        "target_qps": 20.0
    }

    # 2. 生成用户
    np_profiles = generate_user_profiles_multimodal(viz_args["num_users"])
    
    # ==================== 3. 运行 Baseline ====================
    exp_base = VisualExperiment(**viz_args)
    exp_base.user_profiles = np_profiles
    # 🔥【关键修复】强制指定模型名称，覆盖自动检测的 "unknown"
    exp_base.model_name = TARGET_MODEL_NAME 
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task(description=f"Running Baseline ({TARGET_MODEL_NAME})...", total=None)
        await exp_base.run_experiment_poisson("baseline", np_profiles)
    
    baseline_records = exp_base.recordings
    console.print(f"✅ Baseline captured: {len(baseline_records)} tokens")

    # ==================== 4. 运行 Ours ====================
    await asyncio.sleep(2)
    
    exp_ours = VisualExperiment(**viz_args)
    exp_ours.user_profiles = np_profiles
    # 🔥【关键修复】同样强制指定
    exp_ours.model_name = TARGET_MODEL_NAME
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task(description=f"Running Network-Aware ({TARGET_MODEL_NAME})...", total=None)
        await exp_ours.run_experiment_poisson("network_aware", np_profiles)
        
    ours_records = exp_ours.recordings
    console.print(f"✅ Ours captured: {len(ours_records)} tokens")


    # 计算最长持续时间
    max_duration = 0
    if baseline_records: max_duration = max(max_duration, max(r.time_offset for r in baseline_records))
    if ours_records: max_duration = max(max_duration, max(r.time_offset for r in ours_records))

    # 5. 开始倒计时
    for i in range(3, 0, -1):
        console.print(f"[bold yellow]Video starting in {i}...[/bold yellow]")
        time.sleep(1)

    # 6. 播放对比动画
    run_playback(baseline_records, ours_records, max_duration)

    console.print("[bold green]🎬 Demo Finished![/bold green]")

if __name__ == "__main__":
    asyncio.run(main())