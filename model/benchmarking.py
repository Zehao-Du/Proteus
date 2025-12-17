import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
import os

# ================= 配置区域 =================
DATA_PATH = "../data/train_data_congestion.csv"
MODEL_PATH = "best_lstm_grid_search.pth"  # 必须存在这个文件

# 参数必须与训练时完全一致
INPUT_SEQ_LEN = 10   # 输入过去 10 个点
PRED_SEQ_LEN = 10    # 预测未来 10 个点
HIDDEN_SIZE = 256
NUM_LAYERS = 2
TEST_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==========================================

# ================= 1. LSTM 模型定义 (必须一致) =================
class MultiStepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_len):
        super(MultiStepLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_len) 
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ================= 2. 数据处理与加载 =================
def load_and_process_data(path):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    
    # --- 特征工程 (保持一致) ---
    df['avg_rtt_us'] = df['avg_rtt_us'].replace(0, np.nan)
    df.loc[df['avg_rtt_us'] < 10, 'avg_rtt_us'] = np.nan 
    df['avg_rtt_us'] = df['avg_rtt_us'].ffill().bfill()
    
    # Log 变换
    df['log_rtt'] = np.log1p(df['avg_rtt_us']) 
    df['rtt_diff'] = df['log_rtt'].diff().fillna(0) 
    
    feature_cols = ['log_rtt', 'p95_rtt_us', 'avg_cwnd', 'throughput_bps', 'retrans_count', 'rolling_avg_rtt', 'rtt_diff']
    target_col = 'log_rtt'

    data_X = df[feature_cols].values
    data_y = df[target_col].values.reshape(-1, 1)

    # 划分训练/测试集
    split_idx = int(len(df) * (1 - TEST_SPLIT))
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Fit on Train
    X_train_raw = data_X[:split_idx]
    y_train_raw = data_y[:split_idx]
    X_test_raw = data_X[split_idx:]
    y_test_raw = data_y[split_idx:]
    
    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    y_train_scaled = scaler_y.fit_transform(y_train_raw)
    
    X_test_scaled = scaler_X.transform(X_test_raw)
    y_test_scaled = scaler_y.transform(y_test_raw)
    
    print(f"Train samples: {len(X_train_scaled)}, Test samples: {len(X_test_scaled)}")
    return (X_train_scaled, y_train_scaled), (X_test_scaled, y_test_scaled), (scaler_X, scaler_y)

def create_sequences(X, y, input_len, pred_len):
    """
    生成时间序列样本。
    LSTM 输入: [Samples, 10, Features]
    XGBoost 输入: 将 [Samples, 10, Features] 展平为 [Samples, 10*Features]
    """
    xs, ys = [], []
    for i in range(len(X) - input_len - pred_len + 1):
        xs.append(X[i : i + input_len])
        ys.append(y[i + input_len : i + input_len + pred_len])
    
    X_seq = np.array(xs)
    y_seq = np.array(ys).squeeze()
    return X_seq, y_seq

# ================= 3. 模型执行逻辑 =================

def run_lstm_inference(X_test_seq, input_dim):
    """加载权重并推理"""
    print("\n>>> [LSTM] Loading weights and running inference...")
    
    model = MultiStepLSTM(input_dim, HIDDEN_SIZE, NUM_LAYERS, PRED_SEQ_LEN).to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Weight file '{MODEL_PATH}' not found! Please run training script first.")
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    X_tensor = torch.FloatTensor(X_test_seq).to(DEVICE)
    
    with torch.no_grad():
        # Output shape: [Samples, 10]
        preds_scaled = model(X_tensor).cpu().numpy()
        
    return preds_scaled

def run_xgboost_benchmark(X_train_seq, y_train_seq, X_test_seq):
    """
    训练 XGBoost 并推理。
    为了公平对比，XGBoost 应该也能预测未来 10 步。
    我们使用 MultiOutputRegressor 包装 XGBRegressor。
    输入数据需要展平：(Samples, 10, Feats) -> (Samples, 10*Feats)
    """
    print("\n>>> [XGBoost] Training GBDT model...")
    
    # 1. 展平输入数据 (Flatten sequence history)
    # 形状变化: (N, 10, F) -> (N, 10 * F)
    samples, seq_len, feats = X_train_seq.shape
    X_train_flat = X_train_seq.reshape(samples, seq_len * feats)
    
    test_samples, _, _ = X_test_seq.shape
    X_test_flat = X_test_seq.reshape(test_samples, seq_len * feats)
    
    # 2. 定义多输出模型
    # n_estimators=100, max_depth=6 是比较通用的基准参数
    xgb_estimator = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=150, 
        learning_rate=0.05, 
        max_depth=6,
        n_jobs=-1,
        tree_method="hist"  # 加速
    )
    
    # MultiOutputRegressor 允许一次预测多个目标 (未来10步)
    model = MultiOutputRegressor(xgb_estimator)
    
    model.fit(X_train_flat, y_train_seq)
    
    print(">>> [XGBoost] Running inference...")
    preds_scaled = model.predict(X_test_flat)
    
    return preds_scaled

# ================= 4. 评估与可视化 =================

def evaluate_and_plot(lstm_preds_scaled, xgb_preds_scaled, y_true_scaled, scaler_y):
    """
    统一反归一化、计算指标并画图
    """
    print("\n>>> Evaluating results...")
    
    # --- A. 反归一化 (Log Scale -> Real Scale) ---
    # Scaler 是针对单列 fit 的，需要 reshape 才能 inverse
    def inverse_transform_seq(data_scaled):
        # input: [Samples, 10]
        res = np.zeros_like(data_scaled)
        for i in range(data_scaled.shape[1]):
            col_data = data_scaled[:, i].reshape(-1, 1)
            res[:, i] = scaler_y.inverse_transform(col_data).flatten()
        return np.expm1(res) # Inverse Log (expm1)

    lstm_real = inverse_transform_seq(lstm_preds_scaled)
    xgb_real = inverse_transform_seq(xgb_preds_scaled)
    y_true_real = inverse_transform_seq(y_true_scaled)
    
    # --- B. 选取第 10 步 (Step +10) 进行核心对比 ---
    step_idx = PRED_SEQ_LEN - 1 # Index 9 represents Step 10
    
    lstm_final = lstm_real[:, step_idx]
    xgb_final = xgb_real[:, step_idx]
    truth_final = y_true_real[:, step_idx]
    
    # --- C. 计算指标 ---
    def calc_metrics(name, pred, true):
        mae = mean_absolute_error(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        
        # Dangerous Error: 真实值 > 预测值 (低估了RTT，以为路况很好，实际拥塞)
        residuals = true - pred
        under_errors = residuals[residuals > 0]
        mae_under = np.mean(under_errors) if len(under_errors) > 0 else 0
        
        print(f"[{name}] MAE: {mae:.2f} | RMSE: {rmse:.2f} | Danger MAE (Under-est): {mae_under:.2f}")
        return mae, mae_under

    print("-" * 60)
    calc_metrics("XGBoost", xgb_final, truth_final)
    calc_metrics("LSTM   ", lstm_final, truth_final)
    print("-" * 60)

    # --- D. 画图 ---
    plt.figure(figsize=(14, 7))
    
    # 只画前 400 个点以看清细节
    limit = 400
    x_axis = range(limit)
    
    plt.plot(x_axis, truth_final[:limit], label='Ground Truth', color='black', alpha=0.5, linewidth=2)
    plt.plot(x_axis, xgb_final[:limit], label='XGBoost', color='blue', linestyle='--', alpha=0.8)
    plt.plot(x_axis, lstm_final[:limit], label='LSTM (Pre-trained)', color='red', linestyle='-.', alpha=0.9)
    
    plt.title(f"Comparison: LSTM vs XGBoost (Predicting Step +{PRED_SEQ_LEN})")
    plt.xlabel("Test Samples (Time)")
    plt.ylabel("RTT (us)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("benchmark_comparison.png")
    print("\nGraph saved to 'benchmark_comparison.png'")
    plt.show()

# ================= 主程序 =================
if __name__ == "__main__":
    # 1. 准备数据
    (X_train, y_train), (X_test, y_test), scalers = load_and_process_data(DATA_PATH)
    
    # 生成序列数据 (样本数, 10, 特征数)
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, INPUT_SEQ_LEN, PRED_SEQ_LEN)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, INPUT_SEQ_LEN, PRED_SEQ_LEN)
    
    # 2. LSTM 推理 (Load Weights)
    input_dim = X_train.shape[1]
    lstm_preds = run_lstm_inference(X_test_seq, input_dim)
    
    # 3. XGBoost 训练与推理 (Flatten Data)
    xgb_preds = run_xgboost_benchmark(X_train_seq, y_train_seq, X_test_seq)
    
    # 4. 对比评估
    evaluate_and_plot(lstm_preds, xgb_preds, y_test_seq, scalers[1])