#!/usr/bin/env python3
import time
import collections
import numpy as np
import torch
import torch.nn as nn
from bcc import BPF
from collections import deque
import random
import os
import copy
from sklearn.preprocessing import StandardScaler

# ================= 配置与超参数 =================
INTERFACE = "eth0"  # 监听的网卡，虽然 kprobe 是内核级的，但逻辑上我们关注该网卡流量
POLL_INTERVAL = 0.05  # 50ms 采样一次
WINDOW_SIZE = 10      # 滚动窗口统计特征
SEQ_LEN = 10          # LSTM 输入序列长度
PRED_LEN = 10         # LSTM 预测步长
HIDDEN_SIZE = 256
NUM_LAYERS = 2

# 在线学习参数
WARMUP_STEPS = 500    # 前 500 个点(约25秒)只收集数据，用于拟合 Scaler 和初始化
UPDATE_INTERVAL = 10  # 每 10 个数据点训练一次
BATCH_SIZE = 32
MEMORY_SIZE = 1000
ONLINE_LR = 0.001     # 在线微调学习率

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= eBPF 内核代码 (保持不变) =================
BPF_PROGRAM = r"""
#include <uapi/linux/ptrace.h>
#include <linux/types.h>
#include <net/sock.h>
#include <bcc/proto.h>
#include <linux/tcp.h>
#include <linux/skbuff.h>

BPF_PERF_OUTPUT(rtt_events);
BPF_PERF_OUTPUT(retrans_events);

struct rtt_data_t {
    u32 rtt;
    u32 cwnd;
    u32 len;
};

struct retrans_data_t { u32 dummy; };

int trace_tcp_rcv(struct pt_regs *ctx, struct sock *sk, struct sk_buff *skb)
{
    struct tcp_sock *ts = (struct tcp_sock *)sk;
    u32 srtt = ts->srtt_us >> 3;
    if (srtt == 0) return 0;

    struct rtt_data_t data = {};
    data.rtt = srtt;
    data.cwnd = ts->snd_cwnd;
    if (skb) data.len = skb->len;
    else data.len = 0;

    rtt_events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}

int trace_retransmit(struct pt_regs *ctx, struct sock *sk)
{
    struct retrans_data_t data = {};
    retrans_events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
"""

# ================= LSTM 模型定义 =================
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

class AsymmetricMSELoss(nn.Module):
    def __init__(self, penalty=10.0): 
        super().__init__()
        self.penalty = penalty

    def forward(self, pred, target):
        error = target - pred
        # 惩罚低估 (预测 < 真实)
        loss = torch.where(error > 0, error**2 * self.penalty, error**2)
        return torch.mean(loss)

# ================= 在线学习 Agent =================
class OnlineLSTMAgent:
    def __init__(self, input_size):
        self.device = DEVICE
        self.model = MultiStepLSTM(input_size, HIDDEN_SIZE, NUM_LAYERS, PRED_LEN).to(self.device)
        self.criterion = AsymmetricMSELoss(penalty=10.0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=ONLINE_LR)
        self.memory = deque(maxlen=MEMORY_SIZE)
        
        # 尝试加载预训练权重
        if os.path.exists("best_lstm_grid_search.pth"):
            try:
                # 注意：如果特征数量不一致，加载会失败，这里做一个简单的保护
                state = torch.load("best_lstm_grid_search.pth", map_location=self.device)
                # 检查维度是否匹配（粗略检查）
                if state['lstm.weight_ih_l0'].shape[1] == input_size:
                    self.model.load_state_dict(state)
                    print("✅ Loaded pre-trained model weights.")
                else:
                    print("⚠️ Dimension mismatch in pre-trained model. Starting fresh.")
            except Exception as e:
                print(f"⚠️ Load failed: {e}. Starting fresh.")
        else:
            print("ℹ️ No pre-trained model found. Will learn from scratch.")

    def predict(self, seq_array):
        """
        seq_array: shape (1, seq_len, features)
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(seq_array).to(self.device)
            # 确保有 batch 维度
            if x.ndim == 2: x = x.unsqueeze(0)
            pred = self.model(x)
        self.model.train()
        return pred.cpu().numpy()

    def train_step(self, x_seq, y_true):
        """
        x_seq: (seq_len, features)
        y_true: (pred_len, ) -> 实际上我们主要关心 y_true[-1]
        """
        self.memory.append((x_seq, y_true))
        
        if len(self.memory) < BATCH_SIZE:
            return 0.0
        
        # Experience Replay
        batch = random.sample(self.memory, BATCH_SIZE)
        bx, by = zip(*batch)
        bx = torch.FloatTensor(np.array(bx)).to(self.device)
        by = torch.FloatTensor(np.array(by)).to(self.device)
        
        self.optimizer.zero_grad()
        preds = self.model(bx)
        loss = self.criterion(preds, by)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

# ================= 智能采集与处理核心 =================
class SmartMonitor:
    def __init__(self):
        print("⚡ Initializing eBPF...")
        self.bpf = BPF(text=BPF_PROGRAM)
        self.bpf.attach_kprobe(event="tcp_rcv_established", fn_name="trace_tcp_rcv")
        self.bpf.attach_kprobe(event="tcp_retransmit_skb", fn_name="trace_retransmit")
        
        # eBPF Buffer 回调
        self.bpf["rtt_events"].open_perf_buffer(self._handle_rtt)
        self.bpf["retrans_events"].open_perf_buffer(self._handle_retrans)
        
        # 原始数据暂存
        self.raw_rtt = []
        self.raw_cwnd = []
        self.total_bytes = 0
        self.retrans_count = 0
        
        # 特征工程需要的历史状态
        self.rolling_window = deque(maxlen=WINDOW_SIZE)
        self.prev_log_rtt = 0.0
        
        # 在线学习需要的序列 Buffer
        # 存储处理归一化后的特征向量
        self.input_seq_buffer = deque(maxlen=SEQ_LEN)
        
        # 标签生成 Buffer (存储之前的 features，等待未来的 RTT 来标记它)
        # 格式: (timestamp, feature_seq, raw_rtt_target)
        self.pending_training_data = deque(maxlen=PRED_LEN + 5)
        
        self.agent = None # 稍后初始化
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # 状态机
        self.warmup_data_X = []
        self.warmup_data_y = []
        self.steps = 0
        self.is_ready = False

    def _handle_rtt(self, cpu, data, size):
        event = self.bpf["rtt_events"].event(data)
        self.raw_rtt.append(event.rtt)
        self.raw_cwnd.append(event.cwnd)
        self.total_bytes += event.len

    def _handle_retrans(self, cpu, data, size):
        self.retrans_count += 1

    def _process_interval(self):
        """每 50ms 调用一次，聚合数据并进行处理"""
        # 1. 基础聚合
        if not self.raw_rtt:
            avg_rtt = self.rolling_window[-1]['avg_rtt_us'] if self.rolling_window else 0
            p95_rtt = avg_rtt
            avg_cwnd = 0
        else:
            avg_rtt = np.mean(self.raw_rtt)
            p95_rtt = np.percentile(self.raw_rtt, 95)
            avg_cwnd = np.mean(self.raw_cwnd)
        
        throughput = self.total_bytes / POLL_INTERVAL
        r_count = self.retrans_count
        
        # 2. 构造基础 Metrics 字典
        metrics = {
            'avg_rtt_us': avg_rtt,
            'p95_rtt_us': p95_rtt,
            'avg_cwnd': avg_cwnd,
            'throughput_bps': throughput,
            'retrans_count': r_count
        }
        self.rolling_window.append(metrics)
        
        # 3. 计算高级特征 (Feature Engineering)
        # 必须与训练时的特征顺序完全一致:
        # ['log_rtt', 'p95_rtt_us', 'avg_cwnd', 'throughput_bps', 'retrans_count', 'rolling_avg_rtt', 'rtt_diff']
        
        # 处理 Log RTT
        safe_rtt = max(avg_rtt, 1.0) # 防止 log(0)
        log_rtt = np.log1p(safe_rtt)
        
        # 处理 Rolling Avg
        if len(self.rolling_window) > 0:
            roll_avg = np.mean([m['avg_rtt_us'] for m in self.rolling_window])
        else:
            roll_avg = avg_rtt
            
        # 处理 Diff
        rtt_diff = log_rtt - self.prev_log_rtt
        self.prev_log_rtt = log_rtt
        
        # 组合特征向量 (未归一化)
        feature_vector = np.array([
            log_rtt,
            p95_rtt,
            avg_cwnd,
            throughput,
            r_count,
            roll_avg,
            rtt_diff
        ])
        
        # 4. 状态机逻辑
        self.steps += 1
        
        # === 阶段 A: 热身 (Warmup) ===
        if not self.is_ready:
            print(f"🔥 Warming up: {self.steps}/{WARMUP_STEPS}", end='\r')
            self.warmup_data_X.append(feature_vector)
            self.warmup_data_y.append(log_rtt) # 目标是 log_rtt
            
            if self.steps >= WARMUP_STEPS:
                self._finish_warmup()
            return

        # === 阶段 B: 在线运行 (Online) ===
        
        # B1. 归一化当前特征
        # 注意: reshape(1, -1) 因为 scaler 期望 2D 数组
        feat_scaled = self.scaler_X.transform(feature_vector.reshape(1, -1))[0]
        
        # B2. 加入输入序列 Buffer
        self.input_seq_buffer.append(feat_scaled)
        
        # 只有当序列填满 (10个) 才能进行预测和训练
        if len(self.input_seq_buffer) == SEQ_LEN:
            seq_data = np.array(self.input_seq_buffer) # Shape: (10, 7)
            
            # --- 预测 (Prediction) ---
            # 预测未来第10步的 Log RTT
            pred_scaled = self.agent.predict(seq_data) # Shape: (1, 10)
            
            # 取最后一步预测值，反归一化，转回 RTT
            pred_log_rtt = self.scaler_y.inverse_transform(pred_scaled.reshape(1, -1))[0, -1]
            pred_rtt_us = np.expm1(pred_log_rtt)
            
            # 打印监控
            diff = pred_rtt_us - avg_rtt
            marker = "🔴" if diff > 5000 else ("🟢" if abs(diff) < 1000 else "⚪")
            print(f"Step {self.steps} | Real: {avg_rtt:.0f}us | Pred: {pred_rtt_us:.0f}us | Diff: {diff:+.0f} {marker}")
            
            # --- 训练数据准备 (Label Generation) ---
            # 现在的 seq_data 对应时刻 T-9 到 T。
            # 我们想预测 T+1 到 T+10。
            # 但实际上，我们只有等到 T+10 发生时，才能知道那时的真实值。
            # 所以我们将 (Current Sequence, Current Time) 存入 pending 队列。
            # 当时间流逝，未来的真实值出现时，我们再回过头来训练。
            
            # 这里简化逻辑：我们训练模型预测 T+1 (Next Step)
            # 实际上 MultiStepLSTM 预测的是 T+1...T+10
            # 为了简单起见，我们暂存当前的 input sequence
            
            self.pending_training_data.append({
                'seq': copy.deepcopy(seq_data),
                'wait_steps': PRED_LEN, # 等待10步后才有完整标签
                'future_labels': []
            })
            
            # 检查 Pending 队列，填充标签
            for item in self.pending_training_data:
                if item['wait_steps'] > 0:
                    # 记录当前的真实 Log RTT 作为未来的标签
                    # 注意：这里记录的是 log_rtt (未归一化，稍后统一归一化)
                    item['future_labels'].append(log_rtt)
                    item['wait_steps'] -= 1
            
            # 检查是否有数据已经收集满标签
            if self.pending_training_data and self.pending_training_data[0]['wait_steps'] == 0:
                ready_item = self.pending_training_data.popleft()
                
                # 构造 Label
                y_raw = np.array(ready_item['future_labels'])
                y_scaled = self.scaler_y.transform(y_raw.reshape(-1, 1)).flatten()
                
                # --- 训练 (Training) ---
                if self.steps % UPDATE_INTERVAL == 0:
                    loss = self.agent.train_step(ready_item['seq'], y_scaled)
                    if loss > 0:
                        print(f"   🛠️  Model Updated. Loss: {loss:.4f}")

        # 清理本轮计数器
        self.raw_rtt = []
        self.raw_cwnd = []
        self.total_bytes = 0
        self.retrans_count = 0

    def _finish_warmup(self):
        print("\n✅ Warmup complete. Fitting Scalers and initializing Model...")
        
        # 1. 拟合 Scaler
        X_arr = np.array(self.warmup_data_X)
        y_arr = np.array(self.warmup_data_y).reshape(-1, 1)
        
        self.scaler_X.fit(X_arr)
        self.scaler_y.fit(y_arr)
        
        num_features = X_arr.shape[1]
        print(f"   Features identified: {num_features}")
        
        # 2. 初始化 Agent
        self.agent = OnlineLSTMAgent(num_features)
        
        # 3. 填充 Buffer 以便平滑过渡
        # 将热身数据的最后10个填充进 buffer，避免冷启动等待
        for vec in self.warmup_data_X[-SEQ_LEN:]:
            scaled = self.scaler_X.transform(vec.reshape(1, -1))[0]
            self.input_seq_buffer.append(scaled)
            
        self.is_ready = True
        print("🚀 Online Prediction & Training Started!")

    def run(self):
        try:
            while True:
                start_t = time.time()
                # 1. 轮询 eBPF
                self.bpf.perf_buffer_poll(timeout=10) # ms
                
                # 2. 检查时间间隔是否满足处理要求
                # 简单的 sleep 控制，生产环境可以用 timer
                time.sleep(POLL_INTERVAL) 
                
                # 3. 处理逻辑
                self._process_interval()
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
            # 保存最终模型
            if self.agent:
                torch.save(self.agent.model.state_dict(), "final_online_model.pth")
                print("💾 Model saved to final_online_model.pth")

if __name__ == "__main__":
    monitor = SmartMonitor()
    monitor.run()