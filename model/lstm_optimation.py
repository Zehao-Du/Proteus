import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import os

# ================= 配置区域 =================
DATA_PATH = "../data/train_data_congestion.csv"
INPUT_SEQ_LEN = 10
PRED_SEQ_LEN = 10
TEST_SPLIT = 0.2
EPOCHS = 150
BATCH_SIZE = 256 # GPU 下可以适当增大 Batch Size
LR = 0.005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN_SIZE = 256
NUM_LAYERS = 2
# ===========================================

print(f" usando dispositivo: {DEVICE}")

# --- 优化: 非对称 Loss (搬移到 GPU) ---
class AsymmetricMSELoss(nn.Module):
    def __init__(self, penalty=10.0): 
        super().__init__()
        self.penalty = penalty

    def forward(self, pred, target):
        error = target - pred
        # 重罚低估 (target > pred)
        loss = torch.where(error > 0, error**2 * self.penalty, error**2)
        return torch.mean(loss)

def load_data(path):
    print(f"Loading data from {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")
        
    df = pd.read_csv(path)

    # 1. 强力清洗
    df['avg_rtt_us'] = df['avg_rtt_us'].replace(0, np.nan)
    df.loc[df['avg_rtt_us'] < 10, 'avg_rtt_us'] = np.nan 
    df['avg_rtt_us'] = df['avg_rtt_us'].ffill().bfill()
    
    # 2. 对数变换
    df['log_rtt'] = np.log1p(df['avg_rtt_us']) 

    # 3. 特征工程
    df['rtt_diff'] = df['log_rtt'].diff().fillna(0) 
    
    feature_cols = [
        'log_rtt', 'p95_rtt_us', 'avg_cwnd', 'throughput_bps',
        'retrans_count', 'rolling_avg_rtt', 'rtt_diff'
    ]
    target_col = 'log_rtt'

    data_X = df[feature_cols].values
    data_y = df[target_col].values.reshape(-1, 1)
    timestamps = df['timestamp'].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    split_idx = int(len(df) * (1 - TEST_SPLIT))
    
    X_train = scaler_X.fit_transform(data_X[:split_idx])
    X_test = scaler_X.transform(data_X[split_idx:])
    y_train = scaler_y.fit_transform(data_y[:split_idx])
    y_test = scaler_y.transform(data_y[split_idx:])

    return (X_train, y_train), (X_test, y_test), (scaler_X, scaler_y), timestamps[split_idx:]

def create_multistep_sequences(X, y, input_len, pred_len):
    xs, ys = [], []
    for i in range(len(X) - input_len - pred_len + 1):
        xs.append(X[i : i + input_len])
        ys.append(y[i + input_len : i + input_len + pred_len])
    return np.array(xs), np.array(ys).squeeze()

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
        last_step_out = out[:, -1, :] 
        return self.fc(last_step_out)

def run_experiment(data_pack):
    (X_train, y_train), (X_test, y_test), scalers, _ = data_pack
    input_size = X_train.shape[1]
    
    X_train_seq, y_train_seq = create_multistep_sequences(X_train, y_train, INPUT_SEQ_LEN, PRED_SEQ_LEN)
    X_test_seq, y_test_seq = create_multistep_sequences(X_test, y_test, INPUT_SEQ_LEN, PRED_SEQ_LEN)
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train_seq)), 
                              batch_size=BATCH_SIZE, shuffle=True)
    
    model = MultiStepLSTM(input_size, HIDDEN_SIZE, NUM_LAYERS, PRED_SEQ_LEN).to(DEVICE)
    criterion = AsymmetricMSELoss(penalty=10.0).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    print(f"\n🚀 Training with {DEVICE}...")
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE) # 搬运到 GPU
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss/len(train_loader):.4f}")

    # --- 评估 ---
    model.eval()
    with torch.no_grad():
        # 将测试数据搬运到 GPU 预测，再搬回 CPU
        test_x_tensor = torch.FloatTensor(X_test_seq).to(DEVICE)
        preds_scaled = model(test_x_tensor).cpu().numpy()
    
    # 反归一化
    preds_log = scalers[1].inverse_transform(preds_scaled.reshape(-1, 1)).reshape(preds_scaled.shape)
    y_true_log = scalers[1].inverse_transform(y_test_seq.reshape(-1, 1)).reshape(y_test_seq.shape)
    
    # 反对数
    preds_real = np.expm1(preds_log)
    y_true_real = np.expm1(y_true_log)
    
    # --- 核心指标诊断 ---
    residuals = y_true_real[:, -1] - preds_real[:, -1]
    
    # 低估 (真实 > 预测) -> 危险
    under_mask = residuals > 0
    mae_under = np.mean(residuals[under_mask]) if any(under_mask) else 0
    
    # 高估 (真实 < 预测) -> 安全
    over_mask = residuals < 0
    mae_over = np.mean(np.abs(residuals[over_mask])) if any(over_mask) else 0

    mae_total = mean_absolute_error(y_true_real[:, -1], preds_real[:, -1])
    
    print(f"\n✅ Final Results (Step 10):")
    print(f"   Total MAE: {mae_total:.0f} us (总误差)")
    print(f"   Dangerous Error (Under-predict): {mae_under:.0f} us (越小越好)")
    print(f"   Safe Buffer (Over-predict):     {mae_over:.0f} us (模型防御性体现)")
    
    return preds_real, y_true_real, model

def plot_result(preds, truth):
    limit = 500
    plt.figure(figsize=(15, 8))
    plt.plot(range(limit), truth[:limit, 9], color='gray', alpha=0.5, label='Ground Truth (t+1.0s)')
    plt.plot(range(limit), preds[:limit, 9], color='red', linestyle='--', linewidth=1.5, label=f'GPU-LSTM Pred (t+1.0s)')
    plt.title(f"GPU Optimized LSTM: Log-Transform + Asymmetric Loss")
    plt.xlabel("Time Steps (0.1s)")
    plt.ylabel("RTT (us)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("lstm_gpu_result.png")
    plt.show()

if __name__ == "__main__":
    data_pack = load_data(DATA_PATH)
    preds, truth, model = run_experiment(data_pack)
    torch.save(model.state_dict(), "best_lstm_gpu.pth")
    plot_result(preds, truth)