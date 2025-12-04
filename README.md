# 🚀 TokenFlow: 网络感知的 LLM 智能流控系统

> **Network-Aware LLM Token Pacing System powered by eBPF & AI**

[![eBPF](https://img.shields.io/badge/Linux-eBPF-orange.svg)](https://ebpf.io/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![AI](https://img.shields.io/badge/Model-Isolation%20Forest-green.svg)](https://scikit-learn.org/)
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)

TokenFlow (原 SmartNetDiag) 是一个轻量级、低开销的实时网络诊断与流控系统。它利用 **eBPF (Extended Berkeley Packet Filter)** 技术在 Linux 内核态零拷贝采集 TCP 关键指标（RTT、重传），并结合 **孤立森林 (Isolation Forest)** 和 **GBDT** 算法，实现对网络健康度的实时感知。

更进一步，本项目探索了 **网络感知的 LLM 流控 (Network-Aware Token Pacing)**，利用实时网络健康度预测，动态调节大模型 Token 生成速率，实现“自适应流控”，在网络拥塞时自动降速以避免丢包，在网络通畅时全速生成。

---

## 📂 项目目录结构

```text
SmartNetDiag/
├── 📄 run_experiment.sh   # [一键启动] 完整系统编排 (采集+流量+训练+流控+模拟)
├── 📄 smart_agent.py      # [数据面] eBPF 探针，采集内核 TCP RTT/重传
├── 📄 train_model.py      # [智能面] 训练 Isolation Forest (异常检测) + GBDT (RTT预测)
├── 📄 hint_server.py      # [控制面] HTTP 服务，提供 Token Pacing 速率建议 (新增)
├── 📄 llm_simulator.py    # [应用面] 模拟 LLM Token 生成，演示自适应流控 (新增)
├── 📄 dashboard.py        # [可视化] Streamlit 实时监控看板
├── 📄 plot_pacing_effect.py # [可视化] 绘制 Pacing 效果对比图
├── 📄 chaos_maker.py      # [测试] 网络故障注入工具
├── 📄 ROADMAP.md          # [规划] 项目后续开发路线图与分工
├── 📄 requirements.txt    # Python 依赖库列表
└── 📄 README.md           # 项目说明文档
```

---

## 🚀 快速开始 (Quick Start)

### 1. 完整演示 (End-to-End Demo)

使用一键脚本启动整个系统，包括 eBPF 采集、模型训练、流控服务和 LLM 模拟：

```bash
# ⚠️ 需要 sudo 权限以加载 eBPF 程序
sudo bash run_experiment.sh
```

**观察效果**：
1. 脚本会自动启动各个组件。
2. 当后台注入网络故障时，你会看到 LLM Simulator 的输出速率 (`Rate: ... tps`) 自动下降。
3. 当故障恢复时，速率会自动回升。

### 2. 实时看板 (Dashboard)

在另一个终端启动 Web 看板，查看实时网络状态和 AI 诊断结果：

```bash
streamlit run dashboard.py
```

### 3. 可视化分析

生成 RTT 与 Token Rate 的对比图，验证流控效果：

```bash
python plot_pacing_effect.py
```

---

## 🔮 进阶功能：LLM 自适应流控 (Token Pacing)

本项目不仅仅是监控，还实现了 **网络感知的闭环控制**：

1.  **Hint Server**: (`hint_server.py`) 
    *   读取实时网络数据。
    *   调用 **GBDT 模型** 预测未来网络趋势。
    *   通过 HTTP 接口暴露推荐的 `token_rate`。

2.  **LLM Integration**: (`llm_simulator.py`)
    *   在 Token 生成循环中，周期性查询 Hint Server。
    *   根据推荐速率动态调整发送间隔 (`sleep`)。
    *   **效果**：在网络拥塞时自动“刹车”，防止丢包重传导致的卡顿；在网络通畅时全速生成。

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
pip3 install -r requirements.txt
```

---

## 🗺️ 后续规划 (Roadmap)

我们制定了详细的后续开发计划，包括 GPU 监控、真实 vLLM 集成和 PID 控制算法等。
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
*   **闭环控制**：实现了从“监控”到“控制”的跨越，利用 GBDT 预测实现自适应流控。

---

## 📝 License

此项目仅供计算机网络课程学习与研究使用。

---

### 👨‍💻 作者
*   **姓名**：
*   **专业**：计算机科学与技术
