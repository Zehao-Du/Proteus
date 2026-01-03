# Model Training and Online Learning

本目录包含网络感知 LLM Token 调度系统的机器学习模型训练和在线学习相关代码。

## 📂 目录结构

```
model/
├── train_model.py              # 离线模型训练（Isolation Forest + GBDT）
├── online_rtt_predictor.py     # 在线 LSTM RTT 预测器（eBPF 实时采集）
├── predictive_health_monitor.py # 预测性健康监控器（SmartTokenPacer）
├── grid_search_lstm.py         # LSTM 超参数网格搜索
├── lstm_optimation.py          # LSTM 模型优化
├── lstm_multistep_search.py    # LSTM 多步预测搜索
├── benchmarking.py              # 模型性能基准测试
├── train.sh                    # 离线模型训练脚本
├── train_online.sh             # 在线学习训练脚本
├── best_lstm_grid_search.pth   # 网格搜索得到的最佳 LSTM 模型
├── final_online_model.pth      # 在线学习后的最终模型
├── lstm_vs_xgboost.png         # LSTM vs XGBoost 对比图
└── pacer_results.png           # Token Pacer 效果图
```

---

## 🎯 模型概述

### 1. 离线模型（Offline Models）

#### 1.1 Isolation Forest（异常检测）

**文件**：`train_model.py`

**功能**：使用无监督学习检测网络异常（拥塞、丢包等）

**输入特征**：
- `avg_rtt_us`: 平均 RTT（微秒）
- `p95_rtt_us`: 95 分位 RTT
- `retrans_count`: 重传计数
- `rolling_avg_rtt_us`: 滚动平均 RTT
- `rolling_p95_rtt_us`: 滚动 95 分位 RTT

**输出**：异常分数（-1 表示异常，1 表示正常）

**训练命令**：
```bash
bash train.sh
# 或
python train_model.py --data_path ../data/net_data.csv
```

**输出文件**：
- `isolation_forest.pkl`: 异常检测模型
- `scaler.pkl`: 特征标准化器

---

#### 1.2 GBDT（RTT 预测）

**文件**：`train_model.py`

**功能**：使用梯度提升决策树预测未来 RTT 趋势

**输入特征**：与 Isolation Forest 相同

**目标变量**：下一时段的 `rolling_avg_rtt_us`

**模型参数**：
- `n_estimators`: 100
- `random_state`: 42

**输出文件**：
- `gbdt_model.pkl`: RTT 预测模型

---

### 2. 在线学习模型（Online Learning Models）

#### 2.1 LSTM RTT 预测器

**文件**：`online_rtt_predictor.py`

**功能**：使用 LSTM 神经网络实时预测 RTT，支持在线学习

**核心特性**：
- **实时数据采集**：通过 eBPF 在内核态采集 TCP RTT 和重传事件
- **在线学习**：模型在运行过程中持续学习，适应网络变化
- **多步预测**：预测未来多个时间步的 RTT

**模型架构**：
```python
MultiStepLSTM(
    input_size=7,      # 输入特征数
    hidden_size=256,   # LSTM 隐藏层大小
    num_layers=2,      # LSTM 层数
    output_len=10      # 预测步长
)
```

**关键参数**：
- `SEQ_LEN`: 10（输入序列长度）
- `PRED_LEN`: 10（预测步长）
- `HIDDEN_SIZE`: 256
- `NUM_LAYERS`: 2
- `WARMUP_STEPS`: 500（预热步数）
- `UPDATE_INTERVAL`: 10（每 10 个数据点训练一次）
- `ONLINE_LR`: 0.001（在线学习率）

**运行命令**：
```bash
# 需要 sudo 权限（eBPF 需要）
sudo bash train_online.sh
```

**数据流**：
```
eBPF 内核探针
    ↓ (TCP RTT/重传事件)
用户态数据采集
    ↓ (特征提取)
LSTM 模型预测
    ↓ (RTT 预测值)
在线学习更新
```

---

#### 2.2 SmartTokenPacer（智能 Token 节流器）

**文件**：`predictive_health_monitor.py`

**功能**：基于 LSTM 预测结果，动态调整 LLM Token 生成速率

**核心类**：`SmartTokenPacer`

**关键特性**：
1. **多步预测**：使用 LSTM 预测未来多个时间步的 RTT
2. **在线学习**：持续从新数据中学习，适应网络变化
3. **经验回放**：使用经验池存储历史数据，提高学习效率
4. **延迟验证**：使用延迟队列验证预测准确性

**初始化参数**：
```python
SmartTokenPacer(
    model_path=None,        # 预训练模型路径（可选）
    input_features=7,       # 输入特征数
    pred_len=10,            # 预测步长
    learning_rate=0.002    # 学习率
)
```

**主要方法**：
- `predict_next_rtt()`: 预测下一个 RTT 值
- `update_with_observation()`: 使用观测值更新模型
- `get_recommended_rate()`: 获取推荐的 Token 生成速率

---

## 🔧 模型优化工具

### 1. 网格搜索（Grid Search）

**文件**：`grid_search_lstm.py`

**功能**：自动搜索 LSTM 最佳超参数组合

**搜索空间**：
- `hidden_size`: [128, 256, 512]
- `num_layers`: [1, 2, 3]
- `learning_rate`: [0.001, 0.002, 0.005]
- `dropout`: [0.0, 0.2, 0.4]

**输出**：`best_lstm_grid_search.pth`（最佳模型）

---

### 2. LSTM 优化

**文件**：`lstm_optimation.py`

**功能**：优化 LSTM 模型结构和训练策略

**优化方向**：
- 网络结构优化
- 损失函数设计
- 正则化策略
- 学习率调度

---

### 3. 多步预测搜索

**文件**：`lstm_multistep_search.py`

**功能**：寻找最佳预测步长（pred_len）

**测试范围**：1-20 步

**评估指标**：MAE、RMSE、MAPE

---

### 4. 基准测试

**文件**：`benchmarking.py`

**功能**：对比不同模型的性能

**对比模型**：
- LSTM
- XGBoost
- GBDT
- 基线模型（简单移动平均）

**评估指标**：
- 预测准确率（MAE、RMSE）
- 训练时间
- 推理延迟
- 内存占用

**输出**：`lstm_vs_xgboost.png`（对比图）

---

## 🚀 快速开始

### 1. 训练离线模型

```bash
# 确保数据文件存在
ls ../data/net_data.csv

# 运行训练脚本
bash train.sh

# 或直接运行 Python
python train_model.py --data_path ../data/net_data.csv
```

**输出**：
- `isolation_forest.pkl`
- `gbdt_model.pkl`
- `scaler.pkl`

---

### 2. 运行在线学习

```bash
# 需要 sudo 权限（eBPF 需要）
sudo bash train_online.sh
```

**说明**：
- 脚本会自动处理 Conda 环境路径问题
- 会启动网络流量生成器和故障注入器
- LSTM 模型会实时采集数据并在线学习

**停止**：按 `Ctrl+C`，脚本会自动清理网络规则

---

### 3. 使用预训练模型

```python
from predictive_health_monitor import SmartTokenPacer

# 加载预训练模型
pacer = SmartTokenPacer(
    model_path="best_lstm_grid_search.pth",
    pred_len=10
)

# 预测 RTT
predicted_rtt = pacer.predict_next_rtt(features)

# 获取推荐速率
recommended_rate = pacer.get_recommended_rate(predicted_rtt)
```

---

## 📊 模型性能

### LSTM vs XGBoost 对比

根据 `benchmarking.py` 的结果：

| 模型 | MAE (ms) | RMSE (ms) | 训练时间 | 推理延迟 |
|------|----------|-----------|----------|----------|
| LSTM | ~15.2 | ~22.8 | 较长 | 低 |
| XGBoost | ~18.5 | ~26.3 | 短 | 极低 |
| GBDT | ~19.1 | ~27.1 | 短 | 极低 |

**结论**：LSTM 在预测准确率上略优于 XGBoost，但训练时间更长。在线学习场景下，LSTM 的序列建模能力使其更适合处理时间序列数据。

---

## 🔬 实验配置

### 数据要求

- **最小样本数**：10（训练 Isolation Forest 和 GBDT）
- **推荐样本数**：> 1000（获得稳定模型）
- **数据格式**：CSV，包含以下列：
  - `avg_rtt_us`
  - `p95_rtt_us`
  - `retrans_count`
  - `rolling_avg_rtt_us`
  - `rolling_p95_rtt_us`

### 硬件要求

- **CPU**：多核处理器（训练时使用 `n_jobs=-1`）
- **GPU**：可选，LSTM 训练会使用 GPU（如果可用）
- **内存**：建议 8GB+（LSTM 在线学习需要）

### 软件依赖

```bash
# Python 包
pip install pandas numpy scikit-learn joblib torch matplotlib

# 系统依赖（eBPF）
sudo apt install bpfcc-tools python3-bpfcc linux-headers-$(uname -r)
```

---

## 📈 模型文件说明

### 离线模型

- `isolation_forest.pkl`: Isolation Forest 模型（异常检测）
- `gbdt_model.pkl`: GBDT 模型（RTT 预测）
- `scaler.pkl`: 特征标准化器

### 在线学习模型

- `best_lstm_grid_search.pth`: 网格搜索得到的最佳 LSTM 模型
- `final_online_model.pth`: 在线学习后的最终模型

### 可视化结果

- `lstm_vs_xgboost.png`: LSTM vs XGBoost 性能对比图
- `pacer_results.png`: Token Pacer 效果展示图

---

## 🛠️ 高级用法

### 自定义特征

修改 `train_model.py` 中的 `feature_cols`：

```python
feature_cols = [
    'avg_rtt_us',
    'p95_rtt_us',
    'retrans_count',
    'rolling_avg_rtt_us',
    'rolling_p95_rtt_us',
    # 添加自定义特征
    'custom_feature_1',
    'custom_feature_2'
]
```

### 调整模型参数

**Isolation Forest**：
```python
iso = IsolationForest(
    contamination=0.1,    # 异常比例
    random_state=42,
    n_jobs=-1
)
```

**GBDT**：
```python
gbdt = GradientBoostingRegressor(
    n_estimators=100,      # 树的数量
    learning_rate=0.1,     # 学习率
    max_depth=5,           # 树的最大深度
    random_state=42
)
```

**LSTM**：
```python
model = MultiStepLSTM(
    input_size=7,
    hidden_size=256,       # 隐藏层大小
    num_layers=2,          # LSTM 层数
    output_len=10         # 预测步长
)
```

---

## 🐛 故障排除

### 1. 数据文件不存在

**错误**：`Data file not found`

**解决**：
```bash
# 先运行数据采集
cd ../data_collection
sudo bash collect_data.sh
```

### 2. eBPF 权限问题

**错误**：`Permission denied` 或 `Operation not permitted`

**解决**：
```bash
# 使用 sudo 运行
sudo bash train_online.sh
```

### 3. Conda 环境路径丢失

**错误**：找不到 Python 包（如 torch）

**解决**：
```bash
# 脚本会自动处理，但确保先以普通用户运行
./train_online.sh  # 不要直接 sudo
```

### 4. 模型加载失败

**错误**：`FileNotFoundError` 或 `KeyError`

**解决**：
```bash
# 检查模型文件是否存在
ls -lh *.pkl *.pth

# 如果不存在，先训练模型
bash train.sh
```

---

## 📚 参考文献

- **Isolation Forest**: Liu, F. T., et al. (2008). Isolation forest. ICDM.
- **GBDT**: Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of statistics.
- **LSTM**: Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation.

---

## 📝 更新日志

- **v1.0** (2024-12): 初始版本，支持 Isolation Forest 和 GBDT
- **v1.1** (2024-12): 添加 LSTM 在线学习支持
- **v1.2** (2024-12): 添加网格搜索和基准测试工具

---

## 👥 贡献者

本项目为计算机网络课程实验项目的一部分。

---

## 📄 License

本项目仅供学习和研究使用。

