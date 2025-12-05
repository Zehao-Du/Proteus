#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
import math

# 加载模型
try:
    iso_bundle = joblib.load("../agent/isolation_forest.pkl")
    gbdt_bundle = joblib.load("../agent/gbdt_model.pkl")
    scaler = iso_bundle['scaler']
    iso = iso_bundle['model']
    gbdt = gbdt_bundle['model']
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# 读取数据
df = pd.read_csv("../data/net_data.csv")

# 确保有必要的列
feature_cols = ['avg_rtt_us', 'p95_rtt_us', 'retrans_count', 'rolling_avg_rtt_us', 'rolling_p95_rtt_us']
df = df.dropna(subset=feature_cols).reset_index(drop=True)

if df.empty:
    print("No data to plot.")
    exit()

# 模拟 Pacer 逻辑计算 Token Rate
health_scores = []
token_rates = []

X_scaled = scaler.transform(df[feature_cols])
anomaly_scores = iso.decision_function(X_scaled)
pred_next_rtts = gbdt.predict(X_scaled)

for i in range(len(df)):
    # 1. 计算健康度
    # RTT 越小越健康 (基准 10ms = 10000us)
    # 预测的下一个 RTT 用于前瞻控制
    pred_rtt = pred_next_rtts[i]
    rtt_factor = 1.0 / (1.0 + max(0, pred_rtt) / 50000.0)
    
    # 异常扣分
    anomaly_penalty = 0.5 if anomaly_scores[i] < 0 else 0.0
    
    health = max(0.01, min(1.0, rtt_factor - anomaly_penalty))
    health_scores.append(health)
    
    # 2. 计算 Rate
    # Sigmoid 曲线映射
    k = 5
    factor = 1 / (1 + math.exp(-k * (health - 0.5)))
    base_rate = 5.0
    max_rate = 100.0
    rate = base_rate + factor * (max_rate - base_rate)
    token_rates.append(rate)

# 绘图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# 上图：网络状态 (RTT)
ax1.plot(df['timestamp'], df['avg_rtt_us'] / 1000, label='RTT (ms)', color='blue')
ax1.set_ylabel('Latency (ms)')
ax1.set_title('Network Latency (RTT) & Anomalies')
ax1.grid(True, alpha=0.3)

# 标记异常点
anomalies = df[anomaly_scores < 0]
ax1.scatter(anomalies['timestamp'], anomalies['avg_rtt_us'] / 1000, color='red', label='Anomaly Detected', zorder=5)
ax1.legend()

# 下图：Token Rate 自适应曲线
ax2.plot(df['timestamp'], token_rates, label='Adaptive Token Rate (tokens/s)', color='green', linewidth=2)
ax2.set_ylabel('Token Rate (tps)')
ax2.set_xlabel('Timestamp')
ax2.set_title('Adaptive LLM Pacing Control')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Max Capacity')
ax2.legend()

plt.tight_layout()
plt.savefig("pacing_effect.png")
print("✅ Visualization saved to pacing_effect.png")

