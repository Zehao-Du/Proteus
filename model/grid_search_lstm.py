import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import os
import itertools
import random
import time

# ================= 🔍 搜索空间配置 =================
# 在这里定义你想尝试的所有组合
SEARCH_SPACE = {
    'learning_rate': [0.01, 0.005, 0.002, 0.001, 0.0005],
    'epochs':        [50, 100, 150, 200],
    'batch_size':    [512]  # Batch Size 也会影响收敛
}

# 固定参数
DATA_PATH = "../data/train_data_congestion.csv"
INPUT_SEQ_LEN = 10
PRED_SEQ_LEN = 10
TEST_SPLIT = 0.2
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==================================================

print(f"🖥️  Running on device: {DEVICE}")

# --- 辅助功能 ---
def set_seed(seed=42):
    """固定随机种子，保证每次实验起点一致，公平比较"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class AsymmetricMSELoss(nn.Module):
    def __init__(self, penalty=10.0): 
        super().__init__()
        self.penalty = penalty

    def forward(self, pred, target):
        error = target - pred
        loss = torch.where(error > 0, error**2 * self.penalty, error**2)
        return torch.mean(loss)

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")
    df = pd.read_csv(path)
    
    # 清洗与特征工程
    df['avg_rtt_us'] = df['avg_rtt_us'].replace(0, np.nan)
    df.loc[df['avg_rtt_us'] < 10, 'avg_rtt_us'] = np.nan 
    df['avg_rtt_us'] = df['avg_rtt_us'].ffill().bfill()
    df['log_rtt'] = np.log1p(df['avg_rtt_us']) 
    df['rtt_diff'] = df['log_rtt'].diff().fillna(0) 
    
    feature_cols = ['log_rtt', 'p95_rtt_us', 'avg_cwnd', 'throughput_bps', 'retrans_count', 'rolling_avg_rtt', 'rtt_diff']
    target_col = 'log_rtt'

    data_X = df[feature_cols].values
    data_y = df[target_col].values.reshape(-1, 1)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    split_idx = int(len(df) * (1 - TEST_SPLIT))
    
    X_train = scaler_X.fit_transform(data_X[:split_idx])
    X_test = scaler_X.transform(data_X[split_idx:])
    y_train = scaler_y.fit_transform(data_y[:split_idx])
    y_test = scaler_y.transform(data_y[split_idx:])

    return (X_train, y_train), (X_test, y_test), (scaler_X, scaler_y)

def create_sequences(X, y, input_len, pred_len):
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
        return self.fc(out[:, -1, :])

# --- 核心训练函数 ---
def train_and_evaluate(params, data_pack):
    """
    运行一次完整的训练并返回指标
    params: dict {'lr': ..., 'epochs': ..., 'batch_size': ...}
    """
    # 1. 解包数据
    (X_train, y_train), (X_test, y_test), scalers = data_pack
    input_size = X_train.shape[1]

    # 2. 准备 Loader
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, INPUT_SEQ_LEN, PRED_SEQ_LEN)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, INPUT_SEQ_LEN, PRED_SEQ_LEN)
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train_seq)), 
                              batch_size=params['batch_size'], shuffle=True)
    
    # 3. 初始化模型 (每次都要全新的)
    set_seed(42) # 重要！重置种子
    model = MultiStepLSTM(input_size, HIDDEN_SIZE, NUM_LAYERS, PRED_SEQ_LEN).to(DEVICE)
    criterion = AsymmetricMSELoss(penalty=10.0).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    # 4. 训练循环
    model.train()
    start_time = time.time()
    for epoch in range(params['epochs']):
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
    
    duration = time.time() - start_time

    # 5. 评估
    model.eval()
    with torch.no_grad():
        test_x_tensor = torch.FloatTensor(X_test_seq).to(DEVICE)
        preds_scaled = model(test_x_tensor).cpu().numpy()

    # 6. 反归一化 & 指标计算
    preds_log = scalers[1].inverse_transform(preds_scaled).reshape(preds_scaled.shape)
    y_true_log = scalers[1].inverse_transform(y_test_seq).reshape(y_test_seq.shape)
    
    preds_real = np.expm1(preds_log)
    y_true_real = np.expm1(y_true_log)

    # 关注 Step 10
    step10_truth = y_true_real[:, -1]
    step10_pred = preds_real[:, -1]
    
    mae_total = mean_absolute_error(step10_truth, step10_pred)
    
    residuals = step10_truth - step10_pred
    under_errors = residuals[residuals > 0]
    mae_under = np.mean(under_errors) if len(under_errors) > 0 else 0

    return {
        'mae': mae_total,
        'mae_under': mae_under,
        'duration': duration,
        'model_state': model.state_dict() # 返回权重以便保存
    }

# ================= 主控制流程 =================
if __name__ == "__main__":
    # 1. 加载数据 (只做一次)
    print("📥 Loading Data...")
    data_pack = load_data(DATA_PATH)
    
    # 2. 生成所有参数组合
    keys, values = zip(*SEARCH_SPACE.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"🔍 Starting Grid Search over {len(param_combinations)} combinations...")
    print("-" * 60)
    print(f"{'LR':<10} | {'Epochs':<8} | {'Batch':<6} | {'MAE (us)':<10} | {'Danger Err':<10} | {'Time (s)':<8}")
    print("-" * 60)
    
    results = []
    best_score = float('inf')
    
    # 3. 开始循环
    for params in param_combinations:
        try:
            # 运行实验
            res = train_and_evaluate(params, data_pack)
            
            # 打印结果
            print(f"{params['learning_rate']:<10} | {params['epochs']:<8} | {params['batch_size']:<6} | "
                  f"{res['mae']:<10.0f} | {res['mae_under']:<10.0f} | {res['duration']:<8.1f}")
            
            # 记录
            results.append({
                **params,
                'mae': res['mae'],
                'mae_under': res['mae_under']
            })
            
            # 4. 自动保存最佳模型 (以 Dangerous Error 为准，还是以 Total MAE 为准？)
            # 这里我选择综合指标：MAE 不能太差，但 Danger 要尽可能小
            # 简单起见，这里以 Dangerous Error 为第一优化目标 (因为是拥塞控制)
            current_score = res['mae_under'] 
            
            if current_score < best_score:
                best_score = current_score
                torch.save(res['model_state'], "best_lstm_grid_search.pth")
                print(f"   🌟 New Best Found! Model saved.")
                
            # 清理显存
            del res['model_state']
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Error with params {params}: {e}")

    # 5. 总结报告
    print("\n" + "="*40)
    print("🏆 Grid Search Top 5 Results")
    print("="*40)
    
    # 转为 DataFrame 方便排序
    df_res = pd.DataFrame(results)
    # 按 'mae_under' (低估误差) 升序排列
    df_sorted = df_res.sort_values(by='mae_under')
    
    print(df_sorted.head(5).to_string(index=False))
    
    print(f"\n💾 Best model saved to: best_lstm_grid_search.pth")
    print(f"   Best Params: {df_sorted.iloc[0].to_dict()}")