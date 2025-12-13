# streamlit run dashboard.py
import streamlit as st
import pandas as pd
import joblib
import time
import requests
import altair as alt
from collections import deque
from datetime import datetime
import os

# ==========================================
# 1. 配置页面
# ==========================================
st.set_page_config(
    page_title="SmartNetDiag 监控中心",
    page_icon="📡",
    layout="wide"
)

st.title("🚀 基于 eBPF + AI 的智能网络诊断系统")

# ==========================================
# 2. 加载 AI 模型
# ==========================================
@st.cache_resource
def load_model_bundle():
    try:
        if os.path.exists("../agent/isolation_forest.pkl"):
            return joblib.load("../agent/isolation_forest.pkl")
        return None
    except:
        return None

bundle = load_model_bundle()
model = bundle["model"] if bundle else None
scaler = bundle["scaler"] if bundle else None

if model is None:
    st.warning("⚠️ 未检测到 'isolation_forest.pkl' 模型文件，AI 诊断功能已禁用 (仅显示原始数据)。")

# ==========================================
# 3. 数据读取函数
# ==========================================
def get_recent_data(window_size=60):
    try:
        # 兼容两种路径：当前目录 或 ../data/ 目录
        if os.path.exists("net_data.csv"):
            df = pd.read_csv("net_data.csv")
        elif os.path.exists("../data/net_data.csv"):
            df = pd.read_csv("../data/net_data.csv")
        else:
            return pd.DataFrame()
        return df.tail(window_size)
    except:
        return pd.DataFrame()

def get_current_token_rate():
    try:
        resp = requests.get("http://localhost:5000/hint", timeout=0.2)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("token_rate", 0.0), data.get("health", 0.0)
    except:
        pass
    return 0.0, 0.0

# ==========================================
# 4. 主循环
# ==========================================

metric_container = st.empty()
chart_container = st.empty()
alert_container = st.empty()

rate_history = deque(maxlen=60)

while True:
    df = get_recent_data(100)
    current_rate, current_health = get_current_token_rate()
    
    now_str = datetime.now().strftime("%H:%M:%S")
    rate_history.append({"timestamp": now_str, "token_rate": current_rate})
    df_rate = pd.DataFrame(rate_history)

    if not df.empty or not df_rate.empty:
        
        latest_rtt = 0
        latest_retrans = 0
        is_anomaly = False
        
        # --- 数据处理 ---
        if not df.empty:
            # 1. 初始化 anomaly 列，防止 crash
            df['anomaly'] = 1 
            
            # 2. 如果有模型，则覆盖进行预测
            if model is not None:
                try:
                    features = df[['avg_rtt_us', 'p95_rtt_us', 'retrans_count', 'rolling_avg_rtt_us', 'rolling_p95_rtt_us']]
                    if scaler:
                        features_scaled = scaler.transform(features)
                    else:
                        features_scaled = features
                    df['anomaly'] = model.predict(features_scaled)
                except Exception as e:
                    # 如果特征列对不上，保持默认值 1
                    pass

            latest = df.iloc[-1]
            latest_rtt = latest['avg_rtt_us']
            latest_retrans = latest['retrans_count']
            is_anomaly = latest['anomaly'] == -1

        # --- (A) Metrics ---
        with metric_container.container():
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("实时延迟 (RTT)", f"{latest_rtt} us")
            with col2: st.metric("重传次数", f"{latest_retrans}")
            with col3: st.metric("LLM 生成速率", f"{current_rate} tps")
            with col4: 
                if is_anomaly: st.error("🔴 AI: 异常")
                else: st.success("🟢 AI: 健康")

        # --- (B) Alerts ---
        with alert_container.container():
            if is_anomaly:
                st.warning(f"🚨 网络拥塞检测到！速率已限制为 {current_rate} tps")

        # --- (C) Charts ---
        with chart_container.container():
            # 图1: RTT
            if not df.empty:
                chart_data = df.copy()
                base = alt.Chart(chart_data).encode(x=alt.X('timestamp', axis=alt.Axis(labels=False)))
                
                line = base.mark_line().encode(y='avg_rtt_us', color=alt.value("#3366cc"))
                
                # 安全地绘制异常点 (确保列存在)
                points = base.mark_circle(size=60, color='red').encode(
                    y='avg_rtt_us', 
                    tooltip=['avg_rtt_us']
                ).transform_filter(
                    alt.datum.anomaly == -1
                )
                
                st.altair_chart(line + points, use_container_width=True)

            # 图2: Rate
            if not df_rate.empty:
                chart_rate = alt.Chart(df_rate).mark_area(
                    line={'color':'purple'},
                    color=alt.Gradient(
                        gradient='linear',
                        stops=[alt.GradientStop(color='purple', offset=0), alt.GradientStop(color='white', offset=1)],
                        x1=1, x2=1, y1=1, y2=0
                    )
                ).encode(
                    x=alt.X('timestamp'),
                    y=alt.Y('token_rate', scale=alt.Scale(domain=[0, 110]))
                ).properties(height=200)
                st.altair_chart(chart_rate, use_container_width=True)

            # --- 图表 3: 重传计数  ---
            if not chart_data.empty:
                chart_loss = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X('timestamp', axis=alt.Axis(title='Time')), # 最后一个图显示时间轴标签
                    y=alt.Y('retrans_count', axis=alt.Axis(title='Retrans Count')),
                    color=alt.value('orange'),
                    tooltip=['timestamp', 'retrans_count']
                ).properties(title="网络重传事件计数 (Packet Loss)", height=150)
                
                st.altair_chart(chart_loss, use_container_width=True)

    time.sleep(1)