# streamlit run dashboard.py
import streamlit as st
import pandas as pd
import time
import requests
import altair as alt
from collections import deque
from datetime import datetime
import os
import numpy as np

# ==========================================
# 1. 配置页面
# ==========================================
st.set_page_config(
    page_title="TokenFlow 算力调度看板",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 eBPF + LSTM: 网络感知 GPU 算力调度监控")

# ==========================================
# 2. 数据读取函数
# ==========================================
def get_recent_data(window_size=100):
    try:
        # 优先读取 v2 采集器生成的数据
        paths = [
            "../data_collection/train_data.csv",
            "train_data.csv",
            "../data/net_data.csv",
            "net_data.csv"
        ]
        df = pd.DataFrame()
        for p in paths:
            if os.path.exists(p):
                df = pd.read_csv(p)
                break
        
        if df.empty:
            return pd.DataFrame()
        return df.tail(window_size)
    except:
        return pd.DataFrame()

def get_hint_info():
    try:
        resp = requests.get("http://localhost:5000/hint", timeout=0.2)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return {}

# ==========================================
# 3. 主循环
# ==========================================

metric_container = st.empty()
chart_container = st.empty()

rate_history = deque(maxlen=60)

while True:
    df = get_recent_data(100)
    hint = get_hint_info()
    
    current_rate = hint.get("token_rate", 0.0)
    health = hint.get("health", 1.0)
    pred_rtt = hint.get("pred_rtt", 0.0)
    
    now_str = datetime.now().strftime("%H:%M:%S")
    rate_history.append({"timestamp": now_str, "token_rate": current_rate, "health": health})
    df_rate = pd.DataFrame(rate_history)

    # --- (A) Metrics 顶部卡片 ---
    with metric_container.container():
        col1, col2, col3, col4 = st.columns(4)
        
        latest_rtt = df['avg_rtt_us'].iloc[-1] if not df.empty else 0
        latest_tput = df['throughput_bps'].iloc[-1] / 1024 if (not df.empty and 'throughput_bps' in df.columns) else 0
        
        with col1: st.metric("预测 RTT (LSTM)", f"{int(pred_rtt)} us", delta=f"{int(pred_rtt - latest_rtt)} us", delta_color="inverse")
        with col2: st.metric("网络吞吐量", f"{latest_tput:.1f} KB/s")
        with col3: st.metric("GPU 算力分配比", f"{int(health * 100)} %")
        with col4: 
            if health > 0.7: st.success("🟢 状态: 健康")
            elif health > 0.4: st.warning("🟡 状态: 拥塞预警")
            else: st.error("🔴 状态: 极度延迟")

    # --- (B) Charts 图表区 ---
    with chart_container.container():
        # 图1: RTT 与 LSTM 预测线
        if not df.empty:
            st.subheader("网络延迟监控 (Real RTT vs LSTM Prediction)")
            
            # 转换数据格式方便绘图
            df_plot = df.copy()
            df_plot['Real_RTT'] = df_plot['avg_rtt_us']
            
            base = alt.Chart(df_plot).encode(x=alt.X('timestamp:T', title="时间"))
            
            line_real = base.mark_line(opacity=0.5).encode(
                y=alt.Y('Real_RTT', title="延迟 (us)"),
                color=alt.value("#3366cc")
            )
            
            # 画出 CWND 趋势线（次轴）
            if 'avg_cwnd' in df.columns:
                line_cwnd = base.mark_line(strokeDash=[5,5]).encode(
                    y='avg_cwnd',
                    color=alt.value("orange")
                )
                st.altair_chart(line_real + line_cwnd, use_container_width=True)
            else:
                st.altair_chart(line_real, use_container_width=True)

        # 图2: 算力分配与 Token 速率
        if not df_rate.empty:
            st.subheader("GPU 算力分配趋势 (Token Pacing)")
            chart_rate = alt.Chart(df_rate).mark_area(
                line={'color':'purple'},
                color=alt.Gradient(
                    gradient='linear',
                    stops=[alt.GradientStop(color='purple', offset=0), alt.GradientStop(color='white', offset=1)],
                    x1=1, x2=1, y1=1, y2=0
                )
            ).encode(
                x=alt.X('timestamp:T', title="时间"),
                y=alt.Y('token_rate', title="Tokens/s", scale=alt.Scale(domain=[0, 110]))
            ).properties(height=250)
            
            st.altair_chart(chart_rate, use_container_width=True)

    time.sleep(1)
