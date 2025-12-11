# 🚀 eBPF-TokenFlow: Network-Aware Token Pacing for Real-Time LLM Streaming

## 📂 项目目录结构

```text
eBPF-TokenFlow/
├── 📁 agent/                    # [智能平面] 模型训练相关
│   ├── train_model.py          # 读取 CSV 数据，训练 Isolation Forest 和 GBDT 模型
│   ├── train.sh                 # 模型训练脚本
│   ├── isolation_forest.pkl    # 训练好的异常检测模型
│   └── gbdt_model.pkl          # 训练好的 RTT 预测模型
├── 📁 data_collection/          # [数据平面] eBPF 数据采集
│   ├── ebpf_collector.py       # eBPF 探针，负责内核数据采集与清洗
│   ├── collect_data.sh         # 数据采集启动脚本
│   └── chaos_maker.py          # [测试工具] 基于 tc 的网络故障注入器
├── 📁 demo/                     # [应用平面] 演示和集成
│   ├── hint_server.py          # Hint Server：提供网络状态和 token_rate 建议
│   ├── llm_simulator.py        # LLM 模拟器：模拟 Token 生成并响应网络状态
│   ├── dashboard.py            # Streamlit 实时监控仪表盘
│   └── run_demo.sh             # 演示启动脚本
├── 📁 pacer/                    # [LLM 侧边车] 自适应节流
│   └── adaptive_token_pacer.py # 按网络状态自适应节流 LLM token 速率
├── 📁 Visualization/            # [分析工具] 数据可视化
│   └── plot_pacing_effect.py   # RTT 与 Token Rate 对比图生成
├── 📁 data/                     # 数据存储
│   ├── net_data.csv            # eBPF 采集的网络数据
│   └── visualize_data.py       # 数据分布可视化脚本
├── 📁 try/                      # 实验性代码
├── 📄 ROADMAP.md               # 项目开发路线图
├── 📄 submit_pr.sh             # PR 提交脚本
└── 📄 README.md                # 项目说明文档（本文件）
```

---

> **Smart Network Diagnostic System powered by eBPF & Isolation Forest**

[![eBPF](https://img.shields.io/badge/Linux-eBPF-orange.svg)](https://ebpf.io/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![AI](https://img.shields.io/badge/Model-Isolation%20Forest-green.svg)](https://scikit-learn.org/)
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)

SmartNetDiag 是一个轻量级、低开销的实时网络诊断系统。它利用 **eBPF (Extended Berkeley Packet Filter)** 技术在 Linux 内核态零拷贝采集 TCP 关键指标（RTT、重传），并结合 **孤立森林 (Isolation Forest)** 无监督学习算法，实现对网络异常（如拥塞、丢包）的实时检测与根因分析。

---

## 🚀 快速开始 (Quick Start)

### 1. 数据采集 (Data Collection)

启动 eBPF 探针采集网络数据：

```bash
cd data_collection
# ⚠️ 需要 sudo 权限以加载 eBPF 程序
sudo bash collect_data.sh
```

数据将保存到 `../data/net_data.csv`。

### 2. 模型训练 (Model Training)

使用采集的数据训练异常检测和预测模型：

```bash
cd agent
bash train.sh
```

训练完成后会生成 `isolation_forest.pkl` 和 `gbdt_model.pkl`。

### 3. 完整演示 (End-to-End Demo)

启动 Hint Server 和 LLM 模拟器进行完整演示：

```bash
cd demo
# ⚠️ 需要 sudo 权限（Hint Server 需要读取数据文件）
sudo bash run_demo.sh
```

**观察效果**：
1. 脚本会自动启动 Hint Server 和 LLM Simulator。
2. 当后台注入网络故障时，你会看到 LLM Simulator 的输出速率 (`Rate: ... tps`) 自动下降。
3. 当故障恢复时，速率会自动回升。

### 4. 实时看板 (Dashboard)

在另一个终端启动 Web 看板，查看实时网络状态和 AI 诊断结果：

```bash
cd demo
streamlit run dashboard.py
```

### 5. 可视化分析

生成 RTT 与 Token Rate 的对比图，验证流控效果：

```bash
cd Visualization
python plot_pacing_effect.py
```

---

## 🔮 核心功能：LLM 自适应流控 (Token Pacing)

本项目不仅仅是监控，还实现了 **网络感知的闭环控制**：

1.  **Hint Server** (`demo/hint_server.py`) 
    *   读取实时网络数据（从 `data/net_data.csv`）。
    *   调用 **Isolation Forest** 进行异常检测。
    *   调用 **GBDT 模型** 预测未来网络趋势。
    *   通过 HTTP 接口 (`/hint`) 暴露推荐的 `token_rate` 和健康度。

2.  **LLM 模拟器** (`demo/llm_simulator.py`)
    *   在 Token 生成循环中，周期性查询 Hint Server。
    *   根据推荐速率动态调整发送间隔 (`sleep`)。
    *   **效果**：在网络拥塞时自动"刹车"，防止丢包重传导致的卡顿；在网络通畅时全速生成。

3.  **自适应节流器** (`pacer/adaptive_token_pacer.py`)
    *   提供独立的 Token 速率控制逻辑。
    *   支持本地 CLI 模式和云端 HTTP Server 模式。
    *   基于 RTT 和重传计数动态调整速率。

---

## 🛠️ 环境搭建 (Installation)

本项目推荐运行在 **Ubuntu 20.04/22.04 LTS** (物理机、虚拟机或 WSL2) 环境下。

### 1. 系统依赖安装 (eBPF 工具链)

eBPF 依赖较新的内核头文件，请确保系统内核版本 >= 5.8。

```bash
# 更新源
sudo apt update

# 安装 BCC 工具链及内核头文件
sudo apt install -y bison flex build-essential libssl-dev libelf-dev zlib1g-dev \
libfl-dev systemtap-sdt-dev clang llvm \
bpfcc-tools python3-bpfcc libbpfcc libbpfcc-dev linux-headers-$(uname -r)
```

### 2. Python 依赖安装

```bash
# 安装项目所需的 Python 库
pip3 install pandas scikit-learn flask streamlit matplotlib joblib requests
```

**主要依赖**：
- `bcc` / `python3-bpfcc`: eBPF 工具链（通过 apt 安装）
- `pandas`: 数据处理
- `scikit-learn`: 机器学习模型（Isolation Forest, GBDT）
- `flask`: Hint Server Web 框架
- `streamlit`: Dashboard Web 框架
- `matplotlib`: 数据可视化
- `joblib`: 模型序列化
- `requests`: HTTP 客户端

---

## ✅ 当前实现状态

### 已实现功能

- ✅ **eBPF 数据采集**：通过 `tcp_rcv_established` 和 `tcp_retransmit_skb` Hook 采集 TCP RTT 和重传事件
- ✅ **异常检测**：使用 Isolation Forest 进行无监督异常检测
- ✅ **RTT 预测**：使用 GBDT 模型预测未来 RTT 趋势
- ✅ **自适应流控**：基于网络状态动态调整 Token 生成速率（使用 Sigmoid 映射）
- ✅ **Hint Server**：HTTP API 提供网络健康度和推荐速率
- ✅ **LLM 模拟器**：模拟 Token 生成并响应网络状态
- ✅ **实时 Dashboard**：Streamlit 界面展示网络指标和异常检测结果
- ✅ **数据可视化**：生成 RTT 与 Token Rate 对比图

### 待实现功能

我们制定了详细的后续开发计划，包括：
- 🔲 GPU 监控和硬件关联
- 🔲 真实 vLLM/Ollama 集成
- 🔲 PID 控制器（替代当前 Sigmoid 映射）
- 🔲 LSTM/Transformer 模型
- 🔲 在线增量学习
- 🔲 Docker 容器化
- 🔲 Redis 存储升级
- 🔲 自动化测试

详情请见 [ROADMAP.md](./ROADMAP.md)。

---

## 📊 实验结果展示

### 1. 数据特征分布 (Data Distribution)
通过 eBPF 采集的数据呈现清晰的 "L" 型分布：
*   **正常流量**：聚集在原点 (低延迟，无重传)。
*   **拥塞异常**：沿 X 轴延伸 (高延迟，无重传)。
*   **丢包异常**：沿 Y 轴延伸 (低延迟，高重传)。

### 2. 实时监控界面
Dashboard 能够毫秒级捕捉网络波动，并标记异常点。

> *(此处可插入你的 Dashboard 截图)*

---

## 🌟 项目亮点 (Highlights)

*   **零侵入性**：基于 eBPF 技术，无需修改内核源码，无需重启应用，性能开销极低。
*   **真实指标**：通过 Hook `tcp_rcv_established` 和 `tcp_retransmit_skb`，获取内核协议栈真实的 RTT 和重传事件，比 Ping 更准确。
*   **增强型特征**：实时输出最小/最大/平均/95 分位 RTT、重传计数以及滚动均值/分位数，既能反映瞬时尖峰，又能平滑趋势。
*   **智能诊断**：摒弃传统的静态阈值报警，使用 **Isolation Forest** 自动学习网络基线，能够适应不同的网络环境。
*   **闭环控制**：实现了从"监控"到"控制"的跨越，利用 GBDT 预测和 Sigmoid 映射实现自适应流控。
*   **模块化设计**：清晰的目录结构，便于扩展和维护。

---

## 📝 License

此项目仅供计算机网络课程学习与研究使用。

---

### 👨‍💻 作者
*   **姓名**：
*   **专业**：计算机科学与技术
