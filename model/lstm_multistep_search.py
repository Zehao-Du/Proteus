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
DATA_PATH = "../data/train_data_congestion.csv" # 请确保路径正确
INPUT_SEQ_LEN = 10   # 输入：看过去 10 个点 (1.0s)
PRED_SEQ_LEN = 10    # 输出：预测未来 10 个点 (1.0s)
TEST_SPLIT = 0.2
EPOCHS = 60          # 稍微增加轮数，让大模型充分收敛
BATCH_SIZE = 32
LR = 0.001

# 超参数搜索空间
GRID_SEARCH_SPACE = [
    {'hidden_size': 64, 'num_layers': 1},
    {'hidden_size': 128, 'num_layers': 2},
    {'hidden_size': 256, 'num_layers': 2},
]
# ===========================================

def load_data(path):
    print(f"Loading data from {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")
        
    df = pd.read_csv(path)

    # --- 1. 数据清洗 (关键优化) ---
    # RTT=0 是物理不可能的 (采样空窗期 artifacts)，会严重误导模型
    # 我们将其视为缺失值，并用上一时刻的有效值填充 (Forward Fill)
    original_len = len(df)
    df['avg_rtt_us'] = df['avg_rtt_us'].replace(0, np.nan)
    df['avg_rtt_us'] = df['avg_rtt_us'].ffill().bfill() # 先前向填充，开头如果缺则后向填充
    print(f"Data cleaning: Handled 0-value artifacts in {original_len} rows.")

    df = df.dropna()

    # --- 2. 特征工程 ---
    # 增加差分特征 (Gradient)，帮助模型感知“正在变快”还是“正在变慢”
    df['rtt_diff'] = df['avg_rtt_us'].diff().fillna(0)
    
    # 原始特征用于输入
    feature_cols = [
        'avg_rtt_us', 'p95_rtt_us', 'avg_cwnd', 'throughput_bps',
        'retrans_count', 'rolling_avg_rtt', 'rtt_diff'
    ]
    # 预测目标：未来的 avg_rtt_us
    target_col = 'avg_rtt_us' 

    data_X = df[feature_cols].values
    data_y = df[target_col].values.reshape(-1, 1)
    timestamps = df['timestamp'].values

    # 归一化 (Fit on Train, Apply on Test)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Split (按时间切分，不打乱)
    split_idx = int(len(df) * (1 - TEST_SPLIT))
    
    X_train_raw = data_X[:split_idx]
    X_test_raw = data_X[split_idx:]
    y_train_raw = data_y[:split_idx]
    y_test_raw = data_y[split_idx:]
    
    X_train = scaler_X.fit_transform(X_train_raw)
    X_test = scaler_X.transform(X_test_raw)
    y_train = scaler_y.fit_transform(y_train_raw)
    y_test = scaler_y.transform(y_test_raw)

    return (X_train, y_train), (X_test, y_test), (scaler_X, scaler_y), timestamps[split_idx:]

def create_multistep_sequences(X, y, input_len, pred_len):
    """
    构造 Seq2Seq 数据:
    Input:  X[t-9 ... t]
    Target: y[t+1 ... t+10]
    """
    xs, ys = [], []
    for i in range(len(X) - input_len - pred_len + 1):
        xs.append(X[i : i + input_len])
        ys.append(y[i + input_len : i + input_len + pred_len])
    
    return np.array(xs), np.array(ys).squeeze()

# ================= 模型定义 =================
class MultiStepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_len):
        super(MultiStepLSTM, self).__init__()
        
        # dropout防止大模型过拟合
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        
        # Prediction Head
        # 将 LSTM 最后的隐状态映射为未来 10 步的预测
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_len) 
        )

    def forward(self, x):
        # x: [batch, seq_len, features]
        out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出作为 Context Vector
        last_step_out = out[:, -1, :] 
        
        # 预测
        predictions = self.fc(last_step_out) 
        return predictions

# ================= 训练与评估流程 =================
def run_experiment(params, data_pack):
    (X_train, y_train), (X_test, y_test), scalers, _ = data_pack
    input_size = X_train.shape[1]
    
    # 构造序列数据
    X_train_seq, y_train_seq = create_multistep_sequences(X_train, y_train, INPUT_SEQ_LEN, PRED_SEQ_LEN)
    X_test_seq, y_test_seq = create_multistep_sequences(X_test, y_test, INPUT_SEQ_LEN, PRED_SEQ_LEN)
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train_seq)), 
                              batch_size=BATCH_SIZE, shuffle=True)
    
    model = MultiStepLSTM(input_size, params['hidden_size'], params['num_layers'], PRED_SEQ_LEN)
    
    # 使用 MSE Loss
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    print(f"\n🚀 Training Config: {params}")
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for bx, by in train_loader:
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
        preds_scaled = model(torch.FloatTensor(X_test_seq)).numpy()
    
    # 反归一化 [samples, 10]
    preds_real = scalers[1].inverse_transform(preds_scaled.reshape(-1, 1)).reshape(preds_scaled.shape)
    y_true_real = scalers[1].inverse_transform(y_test_seq.reshape(-1, 1)).reshape(y_test_seq.shape)
    
    # 计算 MAE 指标
    mae_overall = mean_absolute_error(y_true_real, preds_real)
    mae_step1 = mean_absolute_error(y_true_real[:, 0], preds_real[:, 0])   # t+0.1s
    mae_step10 = mean_absolute_error(y_true_real[:, -1], preds_real[:, -1]) # t+1.0s
    
    print(f"✅ Result: Overall MAE={mae_overall:.0f} | Step1 MAE={mae_step1:.0f} | Step10 MAE={mae_step10:.0f}")
    
    return {
        "model": model,
        "params": params,
        "mae_overall": mae_overall,
        "preds": preds_real,
        "truth": y_true_real
    }

# ================= 可视化 =================
def plot_best_result(result, timestamps):
    preds = result['preds']
    truth = result['truth']
    params = result['params']
    
    # 限制绘图点数，避免太密看不清
    limit = 500 
    start_idx = 0
    
    plt.figure(figsize=(15, 8))
    
    # 1. Ground Truth (Step 10 的真实值)
    # 我们画出 "Step 10 Truth" 即 t+1.0s 时刻真实发生的 RTT
    plt.plot(range(limit), truth[start_idx:start_idx+limit, 9], color='gray', alpha=0.5, label='Ground Truth (Target at t+1.0s)')
    
    # 2. Step 1 Prediction (短期预测 t+0.1s)
    # 为了对比，我们将 Step 1 的预测画出来（通常它很准，贴着真实值）
    # 注意：这里我们画的是 truth[:,0] 对应的预测，为了视觉不乱，这里暂不画 Step 1 的线，只画 Step 10
    
    # 3. Step 10 Prediction (长期预测 t+1.0s)
    # 这是我们最关心的：模型在 t 时刻，能否预测出 t+1.0s 的波峰？
    plt.plot(range(limit), preds[start_idx:start_idx+limit, 9], color='red', linestyle='--', linewidth=1.5,
             label=f'LSTM Pred (t+1.0s) - MAE: {mean_absolute_error(truth[:,9], preds[:,9]):.0f}')

    plt.title(f"Best Model: {params}\nTask: Predict RTT 1.0s into the future (Step 10)")
    plt.xlabel("Time Steps (0.1s units)")
    plt.ylabel("RTT (us)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("lstm_optimized_result.png")
    print("\n📊 Graph saved to lstm_optimized_result.png")
    plt.show()

# ================= 主程序 =================
if __name__ == "__main__":
    try:
        data_pack = load_data(DATA_PATH)
        (_, _, _, timestamps) = data_pack
        
        best_mae = float('inf')
        best_result = None
        
        print(">>> Starting Hyperparameter Search for Multi-step Prediction...")
        
        for params in GRID_SEARCH_SPACE:
            res = run_experiment(params, data_pack)
            if res['mae_overall'] < best_mae:
                best_mae = res['mae_overall']
                best_result = res
                
                # 保存最佳模型
                torch.save(res['model'].state_dict(), "best_lstm_multistep.pth")
                print("💾 Model saved to best_lstm_multistep.pth")
                
        print("\n🏆 All experiments done.")
        print(f"Best Params: {best_result['params']} with MAE: {best_result['mae_overall']:.2f}")
        
        plot_best_result(best_result, timestamps)
        
    except Exception as e:
        print(f"❌ Error: {e}")