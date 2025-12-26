#!/usr/bin/env python3
"""
Multi-User Hint Server with Simulated Network Conditions

支持多用户场景的 Hint Server：
- 为每个用户维护独立的网络模拟器和健康度
- 提供归一化的算力分配比例
- 支持 A/B 模式切换（Baseline vs Network-Aware）

API Endpoints:
- GET  /hint?user_id=1       - 获取用户1的健康度
- GET  /hint                 - 获取归一化后的全局健康度
- GET  /allocations          - 获取所有用户的算力分配
- POST /mode/baseline        - 切换到 Baseline 模式（平均分配）
- POST /mode/network_aware   - 切换到 Network-Aware 模式
- GET  /stats                - 获取统计信息
"""

import sys
import os
import time
import threading
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List
from collections import deque

import numpy as np
from flask import Flask, jsonify, request

# 添加 model 目录
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
from predictive_health_monitor import SmartTokenPacer

app = Flask(__name__)


# ================= 用户网络模拟器 =================

class UserNetworkSimulator:
    """单用户网络模拟器"""
    
    def __init__(self, user_id: int, base_rtt: float, volatility: str = "normal"):
        self.user_id = user_id
        self.base_rtt = base_rtt
        self.volatility = volatility
        
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
        """生成一个时间步的 RTT"""
        self.step_count += 1
        
        # 随机拥塞
        if np.random.random() < self.congestion_prob:
            self.queue_delay = min(200, self.queue_delay + np.random.uniform(20, 50))
        else:
            self.queue_delay = max(0, self.queue_delay - np.random.uniform(5, 15))
        
        noise = np.random.normal(0, self.noise_scale)
        self.current_rtt = max(10, self.base_rtt + self.queue_delay + noise)
        
        return self.current_rtt
    
    def get_max_receive_rate(self) -> float:
        """根据当前 RTT 计算最大接收速率"""
        BASE_CAPACITY = 100
        RTT_SCALE = 100
        return max(5, BASE_CAPACITY / (1 + self.current_rtt / RTT_SCALE))


# ================= 服务器状态 =================

@dataclass
class UserState:
    user_id: int
    current_rtt: float
    health_score: float
    max_receive_rate: float
    allocated_ratio: float  # 分配比例 (0-1)
    prev_log_rtt: float = 0.0


class ServerState:
    def __init__(self):
        # 模式控制
        self.mode = "network_aware"  # "baseline" or "network_aware"
        
        # 用户配置
        self.users: Dict[int, UserNetworkSimulator] = {}
        self.pacers: Dict[int, SmartTokenPacer] = {}
        self.user_states: Dict[int, UserState] = {}
        
        # 统计
        self.total_requests = 0
        self.mode_switches = 0
        
        # 锁
        self.lock = threading.Lock()
        
        # 后台模拟线程
        self.running = True
        self.sim_thread = None
    
    def add_user(self, user_id: int, base_rtt: float, volatility: str):
        """添加用户"""
        self.users[user_id] = UserNetworkSimulator(user_id, base_rtt, volatility)
        self.pacers[user_id] = SmartTokenPacer(input_features=2, pred_len=10)
        self.pacers[user_id].set_scaler(mean=[4.0, 0.0], scale=[1.0, 1.0])
        self.user_states[user_id] = UserState(
            user_id=user_id,
            current_rtt=base_rtt,
            health_score=1.0,
            max_receive_rate=100,
            allocated_ratio=1.0 / len(self.users) if self.users else 1.0
        )
    
    def simulation_loop(self):
        """后台模拟循环"""
        while self.running:
            with self.lock:
                self._update_all_users()
            time.sleep(0.05)  # 50ms 更新一次
    
    def _update_all_users(self):
        """更新所有用户的状态"""
        health_scores = []
        
        for user_id, simulator in self.users.items():
            # 获取 RTT
            rtt = simulator.step()
            
            # 计算健康度
            pacer = self.pacers[user_id]
            state = self.user_states[user_id]
            
            log_rtt = np.log1p(rtt)
            rtt_diff = log_rtt - state.prev_log_rtt
            
            score, pred_rtt = pacer.step([log_rtt, rtt_diff])
            
            # 更新状态
            state.current_rtt = rtt
            state.health_score = score
            state.max_receive_rate = simulator.get_max_receive_rate()
            state.prev_log_rtt = log_rtt
            
            health_scores.append((user_id, score))
        
        # 计算分配比例
        self._calculate_allocations(health_scores)
    
    def _calculate_allocations(self, health_scores: List[tuple]):
        """计算每个用户的算力分配比例"""
        if self.mode == "baseline":
            # Baseline: 平均分配
            ratio = 1.0 / len(health_scores) if health_scores else 1.0
            for user_id, _ in health_scores:
                self.user_states[user_id].allocated_ratio = ratio
        else:
            # Network-Aware: 按健康度分配
            total_health = sum(score for _, score in health_scores)
            if total_health == 0:
                total_health = 1.0
            
            for user_id, score in health_scores:
                self.user_states[user_id].allocated_ratio = score / total_health
    
    def get_global_health(self) -> float:
        """获取归一化的全局健康度（用于单用户场景的 vLLM）
        
        注意：调用者应该已经持有锁，此方法不再获取锁
        """
        if not self.user_states:
            return 1.0
        
        # 返回加权平均健康度
        total_ratio = 0
        weighted_health = 0
        
        for state in self.user_states.values():
            weighted_health += state.health_score * state.allocated_ratio
            total_ratio += state.allocated_ratio
        
        return weighted_health / total_ratio if total_ratio > 0 else 1.0


STATE = ServerState()


# ================= API Endpoints =================

@app.route("/hint", methods=["GET"])
def get_hint():
    """获取健康度（兼容现有 vLLM 接口）"""
    STATE.total_requests += 1
    
    user_id = request.args.get("user_id", type=int)
    
    with STATE.lock:
        if user_id and user_id in STATE.user_states:
            # 返回特定用户的健康度
            state = STATE.user_states[user_id]
            return jsonify({
                "health": state.health_score,
                "token_rate": state.health_score * 100,
                "user_id": user_id,
                "mode": STATE.mode,
                "metrics": {
                    "rtt": int(state.current_rtt),
                    "max_receive_rate": round(state.max_receive_rate, 1),
                    "allocated_ratio": round(state.allocated_ratio, 3)
                }
            })
        else:
            # 返回全局健康度（归一化）
            global_health = STATE.get_global_health()
            return jsonify({
                "health": global_health,
                "token_rate": global_health * 100,
                "mode": STATE.mode,
                "num_users": len(STATE.users)
            })


@app.route("/allocations", methods=["GET"])
def get_allocations():
    """获取所有用户的算力分配"""
    with STATE.lock:
        allocations = {}
        for user_id, state in STATE.user_states.items():
            allocations[user_id] = {
                "health_score": round(state.health_score, 3),
                "allocated_ratio": round(state.allocated_ratio, 3),
                "current_rtt": round(state.current_rtt, 1),
                "max_receive_rate": round(state.max_receive_rate, 1)
            }
        
        return jsonify({
            "mode": STATE.mode,
            "allocations": allocations
        })


@app.route("/mode/baseline", methods=["POST", "GET"])
def set_mode_baseline():
    """切换到 Baseline 模式"""
    with STATE.lock:
        STATE.mode = "baseline"
        STATE.mode_switches += 1
    return jsonify({"status": "ok", "mode": "baseline"})


@app.route("/mode/network_aware", methods=["POST", "GET"])
def set_mode_network_aware():
    """切换到 Network-Aware 模式"""
    with STATE.lock:
        STATE.mode = "network_aware"
        STATE.mode_switches += 1
    return jsonify({"status": "ok", "mode": "network_aware"})


@app.route("/mode/status", methods=["GET"])
def get_mode_status():
    """获取当前模式"""
    return jsonify({
        "mode": STATE.mode,
        "mode_switches": STATE.mode_switches,
        "total_requests": STATE.total_requests
    })


@app.route("/stats", methods=["GET"])
def get_stats():
    """获取统计信息"""
    with STATE.lock:
        user_stats = []
        for user_id, state in STATE.user_states.items():
            simulator = STATE.users[user_id]
            user_stats.append({
                "user_id": user_id,
                "base_rtt": simulator.base_rtt,
                "volatility": simulator.volatility,
                "current_rtt": round(state.current_rtt, 1),
                "health_score": round(state.health_score, 3),
                "allocated_ratio": round(state.allocated_ratio, 3),
                "steps": simulator.step_count
            })
        
        return jsonify({
            "mode": STATE.mode,
            "total_requests": STATE.total_requests,
            "users": user_stats
        })


@app.route("/health", methods=["GET"])
def health_check():
    """健康检查"""
    return jsonify({"status": "healthy"})


# ================= 主程序 =================

def main():
    parser = argparse.ArgumentParser(description="Multi-User Hint Server")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--mode", choices=["baseline", "network_aware"], default="network_aware")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Multi-User Hint Server with Simulated Network")
    print("=" * 60)
    
    # 初始化用户
    STATE.add_user(user_id=1, base_rtt=150, volatility="chaotic")   # 用户1：网络差
    STATE.add_user(user_id=2, base_rtt=30, volatility="stable")     # 用户2：网络好
    
    print(f"\n👥 Users configured:")
    for uid, sim in STATE.users.items():
        print(f"   User {uid}: base_rtt={sim.base_rtt}ms, volatility={sim.volatility}")
    
    # 设置初始模式
    STATE.mode = args.mode
    print(f"\n🎯 Initial mode: {STATE.mode}")
    
    # 启动后台模拟线程
    STATE.sim_thread = threading.Thread(target=STATE.simulation_loop, daemon=True)
    STATE.sim_thread.start()
    print("✅ Network simulation thread started")
    
    print(f"\n📡 API Endpoints:")
    print(f"   GET  /hint              - Get global health")
    print(f"   GET  /hint?user_id=1    - Get user 1 health")
    print(f"   GET  /allocations       - Get all allocations")
    print(f"   POST /mode/baseline     - Switch to baseline mode")
    print(f"   POST /mode/network_aware - Switch to network-aware mode")
    print(f"   GET  /stats             - Get statistics")
    
    print(f"\n🚀 Starting server on port {args.port}...")
    print("=" * 60)
    
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()

