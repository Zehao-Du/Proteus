#!/usr/bin/env python3
"""
Simulated A/B Experiment for Network-Aware Token Scheduling

实验设计：
- A组（Baseline）：平均分配算力，不考虑网络状况
- B组（Network-Aware）：根据网络健康度分配算力

两个用户：
- 用户1：网络差，最大接收能力由 RTT 决定
- 用户2：网络好，最大接收能力由 RTT 决定

核心指标：ETPS = 有效接收的 Token 数 / 时间
"""

import sys
import os
import numpy as np
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt

# 添加 model 目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
from predictive_health_monitor import SmartTokenPacer, NetworkSimulator


# ================= 1. 用户网络模拟器 =================

class UserNetworkSimulator:
    """
    为单个用户模拟网络状况。
    
    核心逻辑：RTT 越高，每秒能接收的 token 越少。
    max_tokens_per_second = BASE_CAPACITY / (1 + RTT / RTT_SCALE)
    """
    
    def __init__(self, user_id: int, base_rtt: float, volatility: str = "normal"):
        """
        Args:
            user_id: 用户标识
            base_rtt: 基础 RTT (ms)，越高网络越差
            volatility: 网络波动性 ("stable", "normal", "chaotic")
        """
        self.user_id = user_id
        self.base_rtt = base_rtt
        self.volatility = volatility
        
        # 状态变量
        self.current_rtt = base_rtt
        self.queue_delay = 0
        self.step_count = 0
        
        # 波动参数
        if volatility == "stable":
            self.noise_scale = 2
            self.congestion_prob = 0.05
        elif volatility == "normal":
            self.noise_scale = 5
            self.congestion_prob = 0.15
        else:  # chaotic
            self.noise_scale = 15
            self.congestion_prob = 0.3
    
    def step(self) -> float:
        """
        模拟一个时间步的 RTT。
        
        Returns:
            当前 RTT (ms)
        """
        self.step_count += 1
        
        # 随机拥塞事件
        if random.random() < self.congestion_prob:
            # 拥塞：队列延迟增加
            self.queue_delay = min(200, self.queue_delay + random.uniform(20, 50))
        else:
            # 恢复：队列排空
            self.queue_delay = max(0, self.queue_delay - random.uniform(5, 15))
        
        # 添加噪声
        noise = np.random.normal(0, self.noise_scale)
        
        # 计算当前 RTT
        self.current_rtt = max(10, self.base_rtt + self.queue_delay + noise)
        
        return self.current_rtt
    
    def get_max_receive_rate(self, rtt: float) -> float:
        """
        根据 RTT 计算用户的最大接收速率。
        
        模型：max_rate = BASE_CAPACITY / (1 + RTT / RTT_SCALE)
        
        例如：
        - RTT = 30ms  -> max_rate ≈ 77 token/s
        - RTT = 100ms -> max_rate ≈ 50 token/s
        - RTT = 300ms -> max_rate ≈ 25 token/s
        """
        BASE_CAPACITY = 100  # 理想网络下的最大速率
        RTT_SCALE = 100      # RTT 对速率的影响因子
        
        max_rate = BASE_CAPACITY / (1 + rtt / RTT_SCALE)
        return max(5, max_rate)  # 最低 5 token/s


# ================= 2. 模拟调度器 =================

class SimulatedScheduler:
    """
    模拟 vLLM 的调度器行为。
    """
    
    def __init__(self, total_budget: float = 100):
        """
        Args:
            total_budget: 每秒总算力预算 (tokens/s)
        """
        self.total_budget = total_budget
    
    def allocate_baseline(self, num_users: int) -> List[float]:
        """
        A组：Baseline 分配策略（平均分配）
        """
        per_user = self.total_budget / num_users
        return [per_user] * num_users
    
    def allocate_network_aware(self, health_scores: List[float]) -> List[float]:
        """
        B组：Network-Aware 分配策略（按健康度分配）
        
        Args:
            health_scores: 每个用户的健康度分数 (0-1)
        
        Returns:
            每个用户的 token 分配量
        """
        # 归一化健康度
        total_health = sum(health_scores)
        if total_health == 0:
            # 如果所有健康度都是0，回退到平均分配
            return self.allocate_baseline(len(health_scores))
        
        # 按健康度比例分配
        allocations = []
        for score in health_scores:
            ratio = score / total_health
            allocations.append(self.total_budget * ratio)
        
        return allocations


# ================= 3. 实验运行器 =================

@dataclass
class StepResult:
    """单步结果"""
    step: int
    user_id: int
    rtt: float
    health_score: float
    allocated_tokens: float
    max_receive_rate: float
    effective_tokens: float  # min(allocated, max_receive)
    wasted_tokens: float     # allocated - effective


@dataclass
class ExperimentResult:
    """实验结果"""
    group: str  # "baseline" or "network_aware"
    total_steps: int
    total_allocated: float
    total_effective: float
    total_wasted: float
    etps: float  # Effective Tokens Per Second
    user_results: dict  # 每个用户的详细结果


class ABExperiment:
    """A/B 实验运行器"""
    
    def __init__(
        self,
        total_budget: float = 100,
        simulation_steps: int = 1000,
        step_duration_ms: float = 10  # 每步代表 10ms
    ):
        self.total_budget = total_budget
        self.simulation_steps = simulation_steps
        self.step_duration_ms = step_duration_ms
        
        # 创建调度器
        self.scheduler = SimulatedScheduler(total_budget)
        
        # 创建用户（网络条件不同）
        self.users = [
            UserNetworkSimulator(user_id=1, base_rtt=150, volatility="chaotic"),  # 用户1：网络差
            UserNetworkSimulator(user_id=2, base_rtt=30, volatility="stable"),    # 用户2：网络好
        ]
        
        # 为每个用户创建 Pacer
        self.pacers = [
            SmartTokenPacer(input_features=2, pred_len=10),
            SmartTokenPacer(input_features=2, pred_len=10),
        ]
        for pacer in self.pacers:
            pacer.set_scaler(mean=[4.0, 0.0], scale=[1.0, 1.0])
        
        # 记录历史
        self.history = {
            'baseline': [],
            'network_aware': []
        }
    
    def _get_health_score(self, pacer: SmartTokenPacer, rtt: float, prev_log_rtt: float) -> Tuple[float, float]:
        """
        使用 SmartTokenPacer 计算健康度分数。
        """
        log_rtt = np.log1p(rtt)
        rtt_diff = log_rtt - prev_log_rtt
        
        score, pred_rtt = pacer.step([log_rtt, rtt_diff])
        
        return score, log_rtt
    
    def run_baseline(self) -> ExperimentResult:
        """
        运行 A组实验：Baseline（平均分配）
        """
        print("\n" + "="*60)
        print("🔴 Running Group A: BASELINE (Equal Allocation)")
        print("="*60)
        
        # 重置用户状态
        for user in self.users:
            user.step_count = 0
            user.queue_delay = 0
        
        results = []
        prev_log_rtts = [0.0] * len(self.users)
        
        # 累计统计
        total_allocated = 0
        total_effective = 0
        total_wasted = 0
        user_stats = {u.user_id: {'allocated': 0, 'effective': 0, 'wasted': 0} for u in self.users}
        
        # 运行模拟
        for step in range(self.simulation_steps):
            # 获取每个用户的 RTT
            rtts = [user.step() for user in self.users]
            
            # Baseline: 平均分配
            allocations = self.scheduler.allocate_baseline(len(self.users))
            
            # 计算每个用户的有效 token
            for i, (user, rtt, alloc) in enumerate(zip(self.users, rtts, allocations)):
                max_rate = user.get_max_receive_rate(rtt)
                
                # 有效 token = min(分配的, 能接收的)
                effective = min(alloc, max_rate)
                wasted = alloc - effective
                
                # 记录结果
                result = StepResult(
                    step=step,
                    user_id=user.user_id,
                    rtt=rtt,
                    health_score=1.0,  # Baseline 不计算健康度
                    allocated_tokens=alloc,
                    max_receive_rate=max_rate,
                    effective_tokens=effective,
                    wasted_tokens=wasted
                )
                results.append(result)
                
                # 累计
                total_allocated += alloc
                total_effective += effective
                total_wasted += wasted
                user_stats[user.user_id]['allocated'] += alloc
                user_stats[user.user_id]['effective'] += effective
                user_stats[user.user_id]['wasted'] += wasted
        
        # 计算 ETPS
        total_time_seconds = self.simulation_steps * self.step_duration_ms / 1000
        etps = total_effective / total_time_seconds
        
        self.history['baseline'] = results
        
        return ExperimentResult(
            group="baseline",
            total_steps=self.simulation_steps,
            total_allocated=total_allocated,
            total_effective=total_effective,
            total_wasted=total_wasted,
            etps=etps,
            user_results=user_stats
        )
    
    def run_network_aware(self) -> ExperimentResult:
        """
        运行 B组实验：Network-Aware（按健康度分配）
        """
        print("\n" + "="*60)
        print("🟢 Running Group B: NETWORK-AWARE (Health-Based Allocation)")
        print("="*60)
        
        # 重置用户和 Pacer 状态
        for user in self.users:
            user.step_count = 0
            user.queue_delay = 0
        
        # 重新创建 Pacer 以清空状态
        self.pacers = [
            SmartTokenPacer(input_features=2, pred_len=10),
            SmartTokenPacer(input_features=2, pred_len=10),
        ]
        for pacer in self.pacers:
            pacer.set_scaler(mean=[4.0, 0.0], scale=[1.0, 1.0])
        
        results = []
        prev_log_rtts = [0.0] * len(self.users)
        
        # 累计统计
        total_allocated = 0
        total_effective = 0
        total_wasted = 0
        user_stats = {u.user_id: {'allocated': 0, 'effective': 0, 'wasted': 0} for u in self.users}
        
        # 运行模拟
        for step in range(self.simulation_steps):
            # 获取每个用户的 RTT 和健康度
            rtts = []
            health_scores = []
            
            for i, (user, pacer) in enumerate(zip(self.users, self.pacers)):
                rtt = user.step()
                rtts.append(rtt)
                
                # 计算健康度
                score, prev_log_rtts[i] = self._get_health_score(pacer, rtt, prev_log_rtts[i])
                health_scores.append(score)
            
            # Network-Aware: 按健康度分配
            allocations = self.scheduler.allocate_network_aware(health_scores)
            
            # 计算每个用户的有效 token
            for i, (user, rtt, alloc, score) in enumerate(zip(self.users, rtts, allocations, health_scores)):
                max_rate = user.get_max_receive_rate(rtt)
                
                # 有效 token = min(分配的, 能接收的)
                effective = min(alloc, max_rate)
                wasted = alloc - effective
                
                # 记录结果
                result = StepResult(
                    step=step,
                    user_id=user.user_id,
                    rtt=rtt,
                    health_score=score,
                    allocated_tokens=alloc,
                    max_receive_rate=max_rate,
                    effective_tokens=effective,
                    wasted_tokens=wasted
                )
                results.append(result)
                
                # 累计
                total_allocated += alloc
                total_effective += effective
                total_wasted += wasted
                user_stats[user.user_id]['allocated'] += alloc
                user_stats[user.user_id]['effective'] += effective
                user_stats[user.user_id]['wasted'] += wasted
        
        # 计算 ETPS
        total_time_seconds = self.simulation_steps * self.step_duration_ms / 1000
        etps = total_effective / total_time_seconds
        
        self.history['network_aware'] = results
        
        return ExperimentResult(
            group="network_aware",
            total_steps=self.simulation_steps,
            total_allocated=total_allocated,
            total_effective=total_effective,
            total_wasted=total_wasted,
            etps=etps,
            user_results=user_stats
        )
    
    def run_full_experiment(self) -> Tuple[ExperimentResult, ExperimentResult]:
        """
        运行完整的 A/B 实验。
        """
        print("\n" + "🚀"*20)
        print("    SIMULATED A/B EXPERIMENT")
        print("🚀"*20)
        print(f"\n📊 Configuration:")
        print(f"   Total Budget: {self.total_budget} tokens/s")
        print(f"   Simulation Steps: {self.simulation_steps}")
        print(f"   Step Duration: {self.step_duration_ms}ms")
        print(f"   Total Time: {self.simulation_steps * self.step_duration_ms / 1000:.1f}s")
        print(f"\n👥 Users:")
        for user in self.users:
            print(f"   User {user.user_id}: base_rtt={user.base_rtt}ms, volatility={user.volatility}")
        
        # 运行两组实验
        baseline_result = self.run_baseline()
        network_aware_result = self.run_network_aware()
        
        # 打印对比结果
        self._print_comparison(baseline_result, network_aware_result)
        
        return baseline_result, network_aware_result
    
    def _print_comparison(self, baseline: ExperimentResult, network_aware: ExperimentResult):
        """打印对比结果"""
        print("\n" + "="*60)
        print("📊 EXPERIMENT RESULTS COMPARISON")
        print("="*60)
        
        print(f"\n🔴 Group A (BASELINE):")
        print(f"   Total Allocated:  {baseline.total_allocated:.0f} tokens")
        print(f"   Total Effective:  {baseline.total_effective:.0f} tokens")
        print(f"   Total Wasted:     {baseline.total_wasted:.0f} tokens ({baseline.total_wasted/baseline.total_allocated*100:.1f}%)")
        print(f"   ETPS:             {baseline.etps:.2f} tokens/s")
        for uid, stats in baseline.user_results.items():
            print(f"   └─ User {uid}: effective={stats['effective']:.0f}, wasted={stats['wasted']:.0f}")
        
        print(f"\n🟢 Group B (NETWORK-AWARE):")
        print(f"   Total Allocated:  {network_aware.total_allocated:.0f} tokens")
        print(f"   Total Effective:  {network_aware.total_effective:.0f} tokens")
        print(f"   Total Wasted:     {network_aware.total_wasted:.0f} tokens ({network_aware.total_wasted/network_aware.total_allocated*100:.1f}%)")
        print(f"   ETPS:             {network_aware.etps:.2f} tokens/s")
        for uid, stats in network_aware.user_results.items():
            print(f"   └─ User {uid}: effective={stats['effective']:.0f}, wasted={stats['wasted']:.0f}")
        
        # 计算提升
        improvement = (network_aware.etps - baseline.etps) / baseline.etps * 100
        waste_reduction = (baseline.total_wasted - network_aware.total_wasted) / baseline.total_wasted * 100 if baseline.total_wasted > 0 else 0
        
        print(f"\n📈 IMPROVEMENT:")
        print(f"   ETPS Improvement:   {improvement:+.2f}%")
        print(f"   Waste Reduction:    {waste_reduction:+.2f}%")
        
        if improvement > 0:
            print(f"\n   ✅ Network-Aware scheduling outperforms Baseline!")
        else:
            print(f"\n   ⚠️ Baseline performs better (unexpected)")
    
    def plot_results(self, save_path: str = "ab_experiment_result.png"):
        """生成可视化结果"""
        print(f"\n📊 Generating visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 准备数据
        baseline_data = self.history['baseline']
        network_aware_data = self.history['network_aware']
        
        steps = list(range(0, self.simulation_steps * 2, 2))  # 每2步取一个点，减少数据量
        
        # 分离用户数据
        def extract_user_data(data, user_id, metric):
            return [d.__dict__[metric] for d in data if d.user_id == user_id][::2]
        
        # 图1: RTT 对比
        ax1 = axes[0, 0]
        ax1.plot(steps, extract_user_data(baseline_data, 1, 'rtt'), 
                 label='User 1 (Poor Network)', color='red', alpha=0.7)
        ax1.plot(steps, extract_user_data(baseline_data, 2, 'rtt'), 
                 label='User 2 (Good Network)', color='green', alpha=0.7)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('RTT (ms)')
        ax1.set_title('Network RTT by User')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 图2: 健康度 (Network-Aware)
        ax2 = axes[0, 1]
        ax2.plot(steps, extract_user_data(network_aware_data, 1, 'health_score'), 
                 label='User 1 Health', color='red', alpha=0.7)
        ax2.plot(steps, extract_user_data(network_aware_data, 2, 'health_score'), 
                 label='User 2 Health', color='green', alpha=0.7)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Health Score')
        ax2.set_title('Health Score (Network-Aware Group)')
        ax2.legend()
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3)
        
        # 图3: Token 分配对比
        ax3 = axes[1, 0]
        x = ['User 1\n(Poor)', 'User 2\n(Good)']
        baseline_alloc = [self.history['baseline'][0].allocated_tokens, 
                          self.history['baseline'][1].allocated_tokens]
        network_aware_alloc = [
            np.mean(extract_user_data(network_aware_data, 1, 'allocated_tokens')),
            np.mean(extract_user_data(network_aware_data, 2, 'allocated_tokens'))
        ]
        
        bar_width = 0.35
        x_pos = np.arange(len(x))
        ax3.bar(x_pos - bar_width/2, baseline_alloc, bar_width, label='Baseline', color='coral')
        ax3.bar(x_pos + bar_width/2, network_aware_alloc, bar_width, label='Network-Aware', color='seagreen')
        ax3.set_ylabel('Avg Tokens/Step')
        ax3.set_title('Token Allocation Strategy')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(x)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 图4: ETPS 对比
        ax4 = axes[1, 1]
        groups = ['Baseline', 'Network-Aware']
        
        # 从 history 计算 ETPS
        total_time = self.simulation_steps * self.step_duration_ms / 1000
        baseline_etps = sum(d.effective_tokens for d in baseline_data) / total_time
        network_aware_etps = sum(d.effective_tokens for d in network_aware_data) / total_time
        
        etps_values = [baseline_etps, network_aware_etps]
        colors = ['coral', 'seagreen']
        bars = ax4.bar(groups, etps_values, color=colors)
        
        # 添加数值标签
        for bar, val in zip(bars, etps_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{val:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 添加改进百分比
        improvement = (network_aware_etps - baseline_etps) / baseline_etps * 100
        ax4.set_ylabel('ETPS (tokens/s)')
        ax4.set_title(f'Effective Tokens Per Second\n(Improvement: {improvement:+.1f}%)')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"✅ Plot saved to {save_path}")
        

# ================= 4. 主程序 =================

if __name__ == "__main__":
    # 设置随机种子以便复现
    random.seed(42)
    np.random.seed(42)
    
    # 创建实验
    experiment = ABExperiment(
        total_budget=100,        # 每秒总共 100 token 的算力
        simulation_steps=1000,   # 模拟 1000 步
        step_duration_ms=10      # 每步 10ms，总共 10 秒
    )
    
    # 运行实验
    baseline_result, network_aware_result = experiment.run_full_experiment()
    
    # 生成可视化
    experiment.plot_results("ab_experiment_result.png")
    
    print("\n" + "="*60)
    print("✅ Experiment Complete!")
    print("="*60)

