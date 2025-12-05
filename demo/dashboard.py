# streamlit run dashboard.py
import streamlit as st
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt
import altair as alt # Streamlit 自带的高性能绘图库

# ==========================================
# 1. 配置页面
# ==========================================
st.set_page_config(
    page_title="SmartNetDiag 监控中心",
    page_icon="📡",
    layout="wide"
)

st.title("🚀 基于 eBPF + AI 的智能网络诊断系统")
st.markdown("### Smart Network Diagnostic System powered by eBPF & Isolation Forest")

# ==========================================
# 2. 加载 AI 模型
# ==========================================
@st.cache_resource
def load_model_bundle():
    try:
        # train_model.py 保存的是一个字典: {"model": model, "scaler": scaler}
        return joblib.load("isolation_forest.pkl")
    except:
        st.error("未找到模型文件！请先运行 train_model.py")
        return None

bundle = load_model_bundle()
model = bundle["model"] if bundle else None
scaler = bundle["scaler"] if bundle else None

# ==========================================
# 3. 实时读取数据函数
# ==========================================
def get_recent_data(window_size=60):
    try:
        # 只读取最后 window_size 行，避免文件太大卡顿
        df = pd.read_csv("net_data.csv")
        return df.tail(window_size)
    except:
        return pd.DataFrame()

# ==========================================
# 4. 页面布局与实时刷新逻辑
# ==========================================

# 创建占位符容器
metric_container = st.empty()
chart_container = st.empty()
alert_container = st.empty()

while True:
    df = get_recent_data(100) # 获取最近100秒数据
    
    if not df.empty and model is not None:
        # --- 数据预处理 ---
        # 注意：这里需要和训练时使用的特征顺序完全一致
        features = df[['avg_rtt_us', 'p95_rtt_us', 'retrans_count', 'rolling_avg_rtt_us', 'rolling_p95_rtt_us']]
        
        # 必须使用训练时的 scaler 进行同样的归一化，否则模型无法识别
        if scaler:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features
        
        # --- AI 推理 ---
        # 1为正常，-1为异常
        predictions = model.predict(features_scaled)
        df['anomaly'] = predictions
        
        # 获取最新的一条数据
        latest = df.iloc[-1]
        latest_rtt = latest['avg_rtt_us']
        latest_retrans = latest['retrans_count']
        is_anomaly = latest['anomaly'] == -1
        
        # --- (A) 顶部指标栏 (Metrics) ---
        with metric_container.container():
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(label="实时延迟 (RTT)", value=f"{latest_rtt} us", delta=None)
            with col2:
                st.metric(label="重传次数 (Retrans)", value=f"{latest_retrans}", delta=None)
            with col3:
                if is_anomaly:
                    st.error("🔴 AI 诊断: 异常")
                else:
                    st.success("🟢 AI 诊断: 健康")

        # --- (B) 报警分析 ---
        with alert_container.container():
            if is_anomaly:
                reason = []
                if latest_rtt > 20000: # 这里的阈值可以根据你的图调整
                    reason.append("链路拥塞 (High Latency)")
                if latest_retrans > 0:
                    reason.append("丢包丢帧 (Packet Loss)")
                
                error_msg = " | ".join(reason) if reason else "未知异常模式"
                st.warning(f"🚨 检测到网络故障! 根因分析: {error_msg}")

        # --- (C) 可视化图表 ---
        with chart_container.container():
            # 颜色映射：正常点用绿，异常点用红
            chart_data = df.copy()
            chart_data['color'] = chart_data['anomaly'].apply(lambda x: 'red' if x == -1 else '#00AA00')
            
            # 使用 Altair 画一个动态折线图
            # 左图：RTT 趋势
            chart_rtt = alt.Chart(chart_data).mark_line().encode(
                x=alt.X('timestamp', axis=alt.Axis(title='Time')),
                y=alt.Y('avg_rtt_us', axis=alt.Axis(title='RTT (us)')),
                color=alt.value("#3366cc")
            ).properties(title="RTT 实时趋势 (最近100秒)")
            
            # 叠加异常点
            points = alt.Chart(chart_data[chart_data['anomaly']==-1]).mark_circle(size=60).encode(
                x='timestamp',
                y='avg_rtt_us',
                color=alt.value('red'),
                tooltip=['avg_rtt_us', 'retrans_count']
            )

            st.altair_chart(chart_rtt + points, use_container_width=True)

            # 下图：重传柱状图
            chart_loss = alt.Chart(chart_data).mark_bar().encode(
                x='timestamp',
                y='retrans_count',
                color=alt.value('orange')
            ).properties(title="重传事件计数")
            
            st.altair_chart(chart_loss, use_container_width=True)

    # 刷新间隔 1 秒
    time.sleep(1)