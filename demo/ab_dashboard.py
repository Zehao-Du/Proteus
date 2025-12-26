#!/usr/bin/env python3
"""
A/B Comparison Dashboard for Network-Aware Token Pacing

Visualizes the difference between:
- Pacing ON: Network-aware GPU scheduling
- Pacing OFF: Baseline (full speed, no adaptation)

Key Metrics:
- ETPS (Effective Tokens Per Second) = 成功渲染的 token 数 / 完整会话时间
- TTFT (Time To First Token)
- Retransmission Rate
- Network Health

Usage:
    streamlit run ab_dashboard.py
"""

import os
import sys
import time
from datetime import datetime
from collections import deque

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import requests

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="TokenFlow A/B 对比看板",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS for beautiful styling
# ============================================================
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --pacing-on-color: #00d26a;
        --pacing-off-color: #ff6b6b;
        --neutral-color: #4a90d9;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: #e94560;
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        margin: 0;
    }
    
    .main-header p {
        color: #a0a0a0;
        margin: 0.5rem 0 0 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1e1e2e, #2a2a3e);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border-left: 4px solid var(--neutral-color);
    }
    
    .metric-card.pacing-on {
        border-left-color: var(--pacing-on-color);
    }
    
    .metric-card.pacing-off {
        border-left-color: var(--pacing-off-color);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-label {
        color: #888;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Improvement badge */
    .improvement-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .improvement-positive {
        background: linear-gradient(135deg, #00d26a, #00a854);
        color: white;
    }
    
    .improvement-negative {
        background: linear-gradient(135deg, #ff6b6b, #ee5a5a);
        color: white;
    }
    
    /* Section headers */
    .section-header {
        background: #1a1a2e;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 3px solid #e94560;
    }
    
    /* Legend styling */
    .legend-item {
        display: inline-flex;
        align-items: center;
        margin-right: 1.5rem;
    }
    
    .legend-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Data Loading Functions
# ============================================================

@st.cache_data(ttl=5)  # Cache for 5 seconds
def load_experiment_data(data_path: str = "ab_results/latest.csv") -> pd.DataFrame:
    """Load experiment results from CSV."""
    paths_to_try = [
        data_path,
        "demo/ab_results/latest.csv",
        "../demo/ab_results/latest.csv",
        os.path.join(os.path.dirname(__file__), "ab_results/latest.csv")
    ]
    
    for path in paths_to_try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df
    
    return pd.DataFrame()


def get_live_hint() -> dict:
    """Get live data from Hint Server."""
    try:
        resp = requests.get("http://localhost:5000/hint", timeout=0.3)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return {}


# ============================================================
# Sidebar
# ============================================================
st.sidebar.markdown("## ⚙️ 控制面板")

# Data source selection
data_source = st.sidebar.radio(
    "数据来源",
    ["📊 实验结果", "🔴 实时监控"],
    index=0
)

# Refresh rate for live mode
if data_source == "🔴 实时监控":
    refresh_rate = st.sidebar.slider("刷新间隔 (秒)", 0.5, 5.0, 1.0)
else:
    refresh_rate = None

# File upload option
uploaded_file = st.sidebar.file_uploader("上传实验数据 (CSV)", type=['csv'])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 指标说明")
st.sidebar.markdown("""
- **ETPS**: 有效吞吐量 (Effective Tokens/Second)
- **TTFT**: 首 Token 延迟 (Time To First Token)
- **Health**: 网络健康度 (0-1)
- **Retrans**: TCP 重传次数
""")

# ============================================================
# Main Header
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>⚡ eBPF-TokenFlow A/B 对比看板</h1>
    <p>网络感知 GPU 算力调度效果验证</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Load Data
# ============================================================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_experiment_data()

# ============================================================
# Experiment Results View
# ============================================================
if data_source == "📊 实验结果":
    
    if df.empty:
        st.warning("⚠️ 没有找到实验数据。请先运行实验：")
        st.code("python demo/ab_experiment.py --sessions 5 --prompt 'Your prompt here'", language="bash")
        st.stop()
    
    # Split data by group
    df_on = df[df['group'] == 'pacing_on']
    df_off = df[df['group'] == 'pacing_off']
    
    # ============================================================
    # Key Metrics Comparison
    # ============================================================
    st.markdown('<div class="section-header"><h3>📊 核心指标对比</h3></div>', unsafe_allow_html=True)
    
    # Calculate statistics
    def calc_stats(data):
        if data.empty:
            return {'avg_etps': 0, 'avg_ttft': 0, 'total_tokens': 0, 
                    'total_errors': 0, 'total_retrans': 0, 'sessions': 0}
        return {
            'avg_etps': data['etps'].mean(),
            'avg_ttft': data['first_token_latency'].mean(),
            'total_tokens': data['successful_tokens'].sum(),
            'total_errors': data['errors'].sum(),
            'total_retrans': data['retransmits'].sum(),
            'sessions': len(data)
        }
    
    stats_on = calc_stats(df_on)
    stats_off = calc_stats(df_off)
    
    # ETPS Improvement calculation
    if stats_off['avg_etps'] > 0:
        etps_improvement = ((stats_on['avg_etps'] - stats_off['avg_etps']) / stats_off['avg_etps']) * 100
    else:
        etps_improvement = 0
    
    # Top row: ETPS comparison
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(145deg, #0a2e1a, #1a4a2e); padding: 1.5rem; border-radius: 12px; text-align: center;">
            <div style="color: #00d26a; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px;">🟢 Pacing ON</div>
            <div style="color: #00d26a; font-size: 3rem; font-weight: bold; font-family: 'JetBrains Mono', monospace;">{stats_on['avg_etps']:.2f}</div>
            <div style="color: #666; font-size: 0.8rem;">ETPS (有效吞吐量)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        improvement_class = "improvement-positive" if etps_improvement >= 0 else "improvement-negative"
        improvement_sign = "+" if etps_improvement >= 0 else ""
        st.markdown(f"""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; padding-top: 2rem;">
            <div style="color: #888; font-size: 0.8rem; margin-bottom: 0.5rem;">ETPS 提升</div>
            <div class="improvement-badge {improvement_class}">{improvement_sign}{etps_improvement:.1f}%</div>
            <div style="color: #888; font-size: 2rem; margin-top: 0.5rem;">{"→" if etps_improvement >= 0 else "←"}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: linear-gradient(145deg, #2e1a1a, #4a2e2e); padding: 1.5rem; border-radius: 12px; text-align: center;">
            <div style="color: #ff6b6b; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px;">🔴 Pacing OFF</div>
            <div style="color: #ff6b6b; font-size: 3rem; font-weight: bold; font-family: 'JetBrains Mono', monospace;">{stats_off['avg_etps']:.2f}</div>
            <div style="color: #666; font-size: 0.8rem;">ETPS (有效吞吐量)</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Secondary metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ttft_diff = stats_off['avg_ttft'] - stats_on['avg_ttft']
        st.metric(
            "⏱️ 平均 TTFT (Pacing ON)",
            f"{stats_on['avg_ttft']:.3f}s",
            f"{ttft_diff:+.3f}s vs OFF",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            "📦 总 Tokens (Pacing ON)",
            f"{stats_on['total_tokens']:,}",
            f"{stats_on['sessions']} sessions"
        )
    
    with col3:
        retrans_diff = stats_off['total_retrans'] - stats_on['total_retrans']
        st.metric(
            "🔄 重传次数 (Pacing ON)",
            f"{stats_on['total_retrans']}",
            f"{retrans_diff:+d} vs OFF",
            delta_color="inverse"
        )
    
    with col4:
        error_diff = stats_off['total_errors'] - stats_on['total_errors']
        st.metric(
            "❌ 错误次数 (Pacing ON)",
            f"{stats_on['total_errors']}",
            f"{error_diff:+d} vs OFF",
            delta_color="inverse"
        )
    
    # ============================================================
    # ETPS Distribution Chart
    # ============================================================
    st.markdown('<div class="section-header"><h3>📈 ETPS 分布对比</h3></div>', unsafe_allow_html=True)
    
    # Prepare data for chart
    df_chart = df[df['group'].isin(['pacing_on', 'pacing_off'])].copy()
    df_chart['group_label'] = df_chart['group'].map({
        'pacing_on': '🟢 Pacing ON',
        'pacing_off': '🔴 Pacing OFF'
    })
    
    # Box plot
    box_chart = alt.Chart(df_chart).mark_boxplot(
        extent='min-max',
        size=50
    ).encode(
        x=alt.X('group_label:N', title='实验组', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('etps:Q', title='ETPS (Tokens/Second)', scale=alt.Scale(zero=False)),
        color=alt.Color('group:N', scale=alt.Scale(
            domain=['pacing_on', 'pacing_off'],
            range=['#00d26a', '#ff6b6b']
        ), legend=None)
    ).properties(height=300)
    
    # Scatter overlay
    scatter_chart = alt.Chart(df_chart).mark_circle(size=80, opacity=0.6).encode(
        x=alt.X('group_label:N'),
        y=alt.Y('etps:Q'),
        color=alt.Color('group:N', scale=alt.Scale(
            domain=['pacing_on', 'pacing_off'],
            range=['#00d26a', '#ff6b6b']
        ), legend=None),
        tooltip=['session_id', 'etps', 'first_token_latency', 'successful_tokens']
    )
    
    st.altair_chart(box_chart + scatter_chart, use_container_width=True)
    
    # ============================================================
    # Session Timeline
    # ============================================================
    st.markdown('<div class="section-header"><h3>📊 Session 时序图</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ETPS over sessions
        line_chart = alt.Chart(df_chart).mark_line(point=True).encode(
            x=alt.X('session_id:O', title='Session #'),
            y=alt.Y('etps:Q', title='ETPS'),
            color=alt.Color('group_label:N', title='组别', scale=alt.Scale(
                domain=['🟢 Pacing ON', '🔴 Pacing OFF'],
                range=['#00d26a', '#ff6b6b']
            )),
            strokeWidth=alt.value(2)
        ).properties(height=250, title='ETPS 变化趋势')
        
        st.altair_chart(line_chart, use_container_width=True)
    
    with col2:
        # TTFT over sessions
        ttft_chart = alt.Chart(df_chart).mark_bar().encode(
            x=alt.X('session_id:O', title='Session #'),
            y=alt.Y('first_token_latency:Q', title='TTFT (秒)'),
            color=alt.Color('group_label:N', title='组别', scale=alt.Scale(
                domain=['🟢 Pacing ON', '🔴 Pacing OFF'],
                range=['#00d26a', '#ff6b6b']
            )),
            xOffset='group_label:N'
        ).properties(height=250, title='首 Token 延迟 (TTFT)')
        
        st.altair_chart(ttft_chart, use_container_width=True)
    
    # ============================================================
    # Health & Network Metrics
    # ============================================================
    st.markdown('<div class="section-header"><h3>🌐 网络健康度分析</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Health distribution
        health_chart = alt.Chart(df_chart).mark_area(
            opacity=0.5,
            interpolate='monotone'
        ).encode(
            x=alt.X('session_id:O', title='Session #'),
            y=alt.Y('avg_health:Q', title='平均健康度', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('group_label:N', legend=None, scale=alt.Scale(
                domain=['🟢 Pacing ON', '🔴 Pacing OFF'],
                range=['#00d26a', '#ff6b6b']
            ))
        ).properties(height=200, title='网络健康度 (Health Factor)')
        
        st.altair_chart(health_chart, use_container_width=True)
    
    with col2:
        # RTT distribution
        rtt_chart = alt.Chart(df_chart).mark_line(point=True).encode(
            x=alt.X('session_id:O', title='Session #'),
            y=alt.Y('avg_rtt:Q', title='平均 RTT (μs)'),
            color=alt.Color('group_label:N', legend=None, scale=alt.Scale(
                domain=['🟢 Pacing ON', '🔴 Pacing OFF'],
                range=['#00d26a', '#ff6b6b']
            ))
        ).properties(height=200, title='网络延迟 (RTT)')
        
        st.altair_chart(rtt_chart, use_container_width=True)
    
    # ============================================================
    # Raw Data Table
    # ============================================================
    with st.expander("📋 查看原始数据"):
        st.dataframe(df_chart[['session_id', 'group_label', 'etps', 'successful_tokens', 
                               'session_duration', 'first_token_latency', 'avg_health', 
                               'avg_rtt', 'retransmits', 'errors']], use_container_width=True)

# ============================================================
# Live Monitoring View
# ============================================================
else:  # Real-time monitoring
    st.markdown('<div class="section-header"><h3>🔴 实时监控模式</h3></div>', unsafe_allow_html=True)
    
    # Create placeholders for live updates
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    # Initialize history
    if 'live_history' not in st.session_state:
        st.session_state.live_history = deque(maxlen=60)
    
    # Live update loop
    hint = get_live_hint()
    
    if hint:
        # Add to history
        st.session_state.live_history.append({
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'health': hint.get('health', 1.0),
            'token_rate': hint.get('token_rate', 0),
            'pred_rtt': hint.get('pred_rtt', 0),
            'rtt': hint.get('metrics', {}).get('rtt', 0),
            'retrans': hint.get('metrics', {}).get('retrans', 0)
        })
        
        with metrics_placeholder.container():
            col1, col2, col3, col4 = st.columns(4)
            
            health = hint.get('health', 1.0)
            token_rate = hint.get('token_rate', 0)
            pred_rtt = hint.get('pred_rtt', 0)
            metrics = hint.get('metrics', {})
            
            with col1:
                st.metric("🎯 GPU 算力分配", f"{int(health * 100)}%")
            with col2:
                st.metric("⚡ Token 速率", f"{token_rate:.1f} tps")
            with col3:
                st.metric("📡 预测 RTT", f"{int(pred_rtt)} μs")
            with col4:
                if health > 0.7:
                    st.success("🟢 网络健康")
                elif health > 0.4:
                    st.warning("🟡 轻度拥塞")
                else:
                    st.error("🔴 严重拥塞")
        
        # Chart
        if len(st.session_state.live_history) > 1:
            df_live = pd.DataFrame(list(st.session_state.live_history))
            
            with chart_placeholder.container():
                col1, col2 = st.columns(2)
                
                with col1:
                    health_line = alt.Chart(df_live).mark_area(
                        line={'color': '#00d26a'},
                        color=alt.Gradient(
                            gradient='linear',
                            stops=[
                                alt.GradientStop(color='#00d26a', offset=0),
                                alt.GradientStop(color='rgba(0,210,106,0.1)', offset=1)
                            ],
                            x1=1, x2=1, y1=1, y2=0
                        )
                    ).encode(
                        x=alt.X('timestamp:N', title='时间'),
                        y=alt.Y('health:Q', title='健康度', scale=alt.Scale(domain=[0, 1]))
                    ).properties(height=200, title='网络健康度实时曲线')
                    
                    st.altair_chart(health_line, use_container_width=True)
                
                with col2:
                    rate_line = alt.Chart(df_live).mark_line(
                        color='#e94560',
                        strokeWidth=2
                    ).encode(
                        x=alt.X('timestamp:N', title='时间'),
                        y=alt.Y('token_rate:Q', title='Tokens/s')
                    ).properties(height=200, title='Token 生成速率')
                    
                    st.altair_chart(rate_line, use_container_width=True)
    else:
        st.warning("⚠️ 无法连接到 Hint Server。请确保服务正在运行：")
        st.code("python demo/hint_server.py", language="bash")
    
    # Auto-refresh
    if refresh_rate:
        time.sleep(refresh_rate)
        st.rerun()

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>eBPF-TokenFlow | 网络感知 GPU 算力调度系统</p>
    <p>💡 提示：使用 <code>python demo/ab_experiment.py</code> 生成实验数据</p>
</div>
""", unsafe_allow_html=True)

